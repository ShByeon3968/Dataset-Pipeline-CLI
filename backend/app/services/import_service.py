"""
Annotated ZIP import service

Supported formats:
  - COCO JSON
      Structure A) images/ + annotations/*.json  (CVAT, Label Studio, etc.)
      Structure B) train/ val/ test/ each with images + *annotations*.json
                   images under split/ or split/images/
                   (Roboflow COCO, CVAT per-split, etc.)
  - YOLO
      data.yaml or classes.txt for class definitions
      labels/*.txt for annotations (cx cy w h normalized)
      Roboflow style (train/images/ + train/labels/) supported

Split detection:
  - Structure B COCO: split_dir name (train / val / valid / test)
  - YOLO: image path prefix (train/ val/ valid/ test/)
  - Images only or no split directories: split = None
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Image, Annotation
from app.services.class_service import get_or_create_class
from app.services.file_handler import (
    save_image_to_storage,
    calculate_md5_bytes,
    calculate_phash,
    get_image_dimensions,
    get_file_format,
    SUPPORTED_FORMATS,
)

BATCH_SIZE = 100

_SPLIT_ALIASES: dict[str, str] = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


def _normalize_split(name: str) -> str | None:
    """Convert directory name to canonical split name (train/val/test) or None."""
    return _SPLIT_ALIASES.get(name.lower())


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_format(names: list[str]) -> str:
    lower = [n.lower() for n in names]
    has_json = any(n.endswith(".json") for n in lower)
    has_yaml = any(Path(n).name.lower() in ("data.yaml", "dataset.yaml") for n in names)
    has_classes_txt = any(Path(n).name.lower() == "classes.txt" for n in names)
    has_label_txt = any(n.lower().endswith(".txt") and "label" in n.lower() for n in names)
    if has_json:
        return "coco"
    if has_yaml or has_classes_txt or has_label_txt:
        return "yolo"
    return "coco"


# ---------------------------------------------------------------------------
# COCO structure helpers
# ---------------------------------------------------------------------------

def _is_image_file(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_FORMATS


def _find_image_prefix(zf: zipfile.ZipFile, base_dir: str) -> str:
    names = zf.namelist()
    sub = (base_dir.rstrip("/") + "/images/").lstrip("/")
    if any(n.startswith(sub) and _is_image_file(n) for n in names):
        return sub
    base = (base_dir.rstrip("/") + "/").lstrip("/")
    return base


def _parse_coco_structure(zf: zipfile.ZipFile) -> list[dict]:
    """
    Returns list of:
      {"json": <zip_path>, "image_prefix": <str>, "split": "train"|"val"|"test"|None}

    Structure B (split directories) is preferred over structure A.
    """
    names = zf.namelist()
    json_files = [n for n in names if n.lower().endswith(".json")]
    splits: list[dict] = []

    # Structure B: top-level split directories containing JSON + images
    split_json: dict[str, str] = {}
    for j in json_files:
        parts = Path(j).parts
        if len(parts) == 2:
            split_dir = parts[0]
            has_imgs = any(
                n.startswith(split_dir + "/") and _is_image_file(n) for n in names
            )
            if has_imgs and split_dir not in split_json:
                split_json[split_dir] = j

    if split_json:
        for split_dir, json_path in split_json.items():
            image_prefix = _find_image_prefix(zf, split_dir)
            splits.append({
                "json": json_path,
                "image_prefix": image_prefix,
                "split": _normalize_split(split_dir),
            })
        return splits

    # Structure A: flat annotations/ + images/
    anno_jsons = [
        j for j in json_files
        if len(Path(j).parts) >= 2 and Path(j).parts[-2].lower() == "annotations"
    ]
    if not anno_jsons:
        anno_jsons = [
            j for j in json_files
            if "annotation" in Path(j).name.lower() and len(Path(j).parts) == 1
        ]
    if not anno_jsons:
        anno_jsons = [j for j in json_files if len(Path(j).parts) == 1]
    if not anno_jsons:
        anno_jsons = json_files[:1]

    for j in anno_jsons:
        parent_parts = Path(j).parts
        if len(parent_parts) >= 2:
            base = "/".join(parent_parts[:-2]) if len(parent_parts) > 2 else ""
            image_prefix = _find_image_prefix(zf, base) if base else "images/"
        else:
            has_images_dir = any(n.startswith("images/") and _is_image_file(n) for n in names)
            image_prefix = "images/" if has_images_dir else ""
        splits.append({"json": j, "image_prefix": image_prefix, "split": None})

    return splits


# ---------------------------------------------------------------------------
# File save helper
# ---------------------------------------------------------------------------

def _save_image_bytes(data: bytes, filename: str, dataset_id: int) -> tuple[str, str]:
    """
    이미지 bytes 를 스토리지(MinIO 또는 로컬)에 저장.
    Returns (local_path_for_metadata, storage_key_for_db).
    """
    return save_image_to_storage(data, filename, dataset_id)


# ---------------------------------------------------------------------------
# COCO importer
# ---------------------------------------------------------------------------

async def _import_coco(
    db: AsyncSession,
    dataset_id: int,
    zf: zipfile.ZipFile,
    existing_hashes: set[str],
) -> dict[str, int]:
    split_entries = _parse_coco_structure(zf)

    added = skipped = errors = 0

    for entry in split_entries:
        json_path: str | None = entry.get("json")
        image_prefix: str = entry.get("image_prefix", "")
        split_value: str | None = entry.get("split")

        if not json_path:
            continue

        try:
            coco: dict[str, Any] = json.loads(zf.read(json_path).decode("utf-8"))
        except Exception:
            errors += 1
            continue

        # Category -> DB class mapping
        categories: dict[int, str] = {
            cat["id"]: cat["name"] for cat in coco.get("categories", [])
        }
        class_map: dict[int, int] = {}
        for coco_id, name in categories.items():
            cls = await get_or_create_class(db, dataset_id, name)
            class_map[coco_id] = cls.id

        lower_map: dict[str, str] = {n.lower(): n for n in zf.namelist()}
        coco_img_id_to_db: dict[int, int] = {}
        coco_img_size: dict[int, tuple[int, int]] = {}
        pending: list[tuple[int, Image]] = []

        for img_info in coco.get("images", []):
            filename: str = img_info.get("file_name", "")
            coco_img_id: int = img_info["id"]
            coco_w: int = img_info.get("width") or 0
            coco_h: int = img_info.get("height") or 0

            stem_name = Path(filename).name
            candidates = [
                image_prefix + filename,
                image_prefix + stem_name,
                filename,
                stem_name,
            ]
            zip_member: str | None = None
            for cp in candidates:
                if cp in zf.namelist():
                    zip_member = cp
                    break
                if cp.lower() in lower_map:
                    zip_member = lower_map[cp.lower()]
                    break

            if zip_member is None:
                errors += 1
                continue

            if Path(zip_member).suffix.lower() not in SUPPORTED_FORMATS:
                errors += 1
                continue

            data = zf.read(zip_member)
            md5 = calculate_md5_bytes(data)

            if md5 in existing_hashes:
                skipped += 1
                coco_img_id_to_db[coco_img_id] = -1
                coco_img_size[coco_img_id] = (coco_w or 1, coco_h or 1)
                continue

            abs_path, rel_path = _save_image_bytes(data, stem_name, dataset_id)

            if coco_w and coco_h:
                w, h = coco_w, coco_h
            else:
                try:
                    w, h = get_image_dimensions(abs_path)
                except Exception:
                    w, h = 1, 1

            phash = ""
            try:
                phash = calculate_phash(abs_path)
            except Exception:
                pass

            img_obj = Image(
                dataset_id=dataset_id,
                filename=Path(abs_path).name,
                filepath=rel_path,
                width=w, height=h,
                format=get_file_format(abs_path),
                file_hash=md5,
                phash=phash,
                split=split_value,
            )
            db.add(img_obj)
            pending.append((coco_img_id, img_obj))
            coco_img_size[coco_img_id] = (w, h)
            existing_hashes.add(md5)
            added += 1

            if len(pending) >= BATCH_SIZE:
                await db.flush()
                for c_id, img in pending:
                    coco_img_id_to_db[c_id] = img.id
                pending.clear()

        if pending:
            await db.flush()
            for c_id, img in pending:
                coco_img_id_to_db[c_id] = img.id
            pending.clear()

        # Annotations
        ann_batch: list[Annotation] = []
        for ann_info in coco.get("annotations", []):
            coco_img_id = ann_info.get("image_id")
            db_img_id = coco_img_id_to_db.get(coco_img_id)
            if db_img_id is None or db_img_id == -1:
                continue

            bbox = ann_info.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            coco_cat_id = ann_info.get("category_id")
            db_class_id = class_map.get(coco_cat_id)

            img_w, img_h = coco_img_size.get(coco_img_id, (1, 1))
            x_min_px, y_min_px, w_px, h_px = (float(v) for v in bbox)

            ann_batch.append(Annotation(
                image_id=db_img_id,
                class_id=db_class_id,
                annotation_type="bbox",
                bbox_x=x_min_px / img_w,
                bbox_y=y_min_px / img_h,
                bbox_w=w_px / img_w,
                bbox_h=h_px / img_h,
            ))

            if len(ann_batch) >= BATCH_SIZE:
                db.add_all(ann_batch)
                await db.flush()
                ann_batch.clear()

        if ann_batch:
            db.add_all(ann_batch)
            await db.flush()

    return {"added": added, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# YOLO importer
# ---------------------------------------------------------------------------

def _load_yolo_classes(zf: zipfile.ZipFile) -> list[str]:
    names_in_zip = zf.namelist()
    for n in names_in_zip:
        if Path(n).name.lower() in ("data.yaml", "dataset.yaml"):
            try:
                import yaml
                raw = yaml.safe_load(zf.read(n).decode("utf-8"))
                if isinstance(raw, dict):
                    cls_names = raw.get("names")
                    if isinstance(cls_names, list):
                        return [str(x) for x in cls_names]
                    if isinstance(cls_names, dict):
                        return [cls_names[k] for k in sorted(cls_names)]
            except Exception:
                pass
    for n in names_in_zip:
        if Path(n).name.lower() == "classes.txt":
            try:
                lines = zf.read(n).decode("utf-8").splitlines()
                return [ln.strip() for ln in lines if ln.strip()]
            except Exception:
                pass
    return []


def _find_label_file(zf: zipfile.ZipFile, img_zip_path: str) -> str | None:
    names_in_zip_set = set(zf.namelist())
    lower_map = {n.lower(): n for n in zf.namelist()}
    stem = Path(img_zip_path).stem
    parts = Path(img_zip_path).parts
    candidates: list[str] = []

    # Roboflow: train/images/img.jpg -> train/labels/img.txt
    if len(parts) >= 3 and parts[-2].lower() == "images":
        split_prefix = "/".join(parts[:-2])
        candidates.append(f"{split_prefix}/labels/{stem}.txt")

    candidates += [
        f"labels/{stem}.txt",
        f"labels/train/{stem}.txt",
        f"labels/val/{stem}.txt",
        f"labels/valid/{stem}.txt",
        f"labels/test/{stem}.txt",
    ]
    if len(parts) >= 2:
        split_prefix = parts[0]
        candidates += [
            f"{split_prefix}/labels/{stem}.txt",
            f"{split_prefix}/{stem}.txt",
        ]
    candidates.append(f"{stem}.txt")

    for c in candidates:
        if c in names_in_zip_set:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _split_from_yolo_path(img_zip_path: str) -> str | None:
    """
    Infer split from YOLO image path.
    train/images/img.jpg   -> "train"
    val/images/img.jpg     -> "val"
    valid/images/img.jpg   -> "val"
    test/images/img.jpg    -> "test"
    images/img.jpg         -> None
    img.jpg                -> None
    """
    parts = Path(img_zip_path).parts
    if len(parts) >= 2:
        # Roboflow: split/images/filename
        if len(parts) >= 3 and parts[-2].lower() == "images":
            return _normalize_split(parts[0])
        # Fallback: top-level directory name
        return _normalize_split(parts[0])
    return None


async def _import_yolo(
    db: AsyncSession,
    dataset_id: int,
    zf: zipfile.ZipFile,
    existing_hashes: set[str],
) -> dict[str, int]:
    class_names = _load_yolo_classes(zf)
    class_db_map: dict[int, int] = {}

    async def resolve_class(yolo_idx: int) -> int | None:
        if yolo_idx in class_db_map:
            return class_db_map[yolo_idx]
        name = class_names[yolo_idx] if yolo_idx < len(class_names) else f"class_{yolo_idx}"
        cls = await get_or_create_class(db, dataset_id, name)
        class_db_map[yolo_idx] = cls.id
        return cls.id

    image_members = [
        n for n in zf.namelist()
        if _is_image_file(n) and not n.startswith("__MACOSX")
    ]

    added = skipped = errors = 0
    pending: list[tuple[str, Image]] = []

    for zip_member in image_members:
        filename = Path(zip_member).name
        if Path(filename).suffix.lower() not in SUPPORTED_FORMATS:
            errors += 1
            continue

        data = zf.read(zip_member)
        md5 = calculate_md5_bytes(data)
        if md5 in existing_hashes:
            skipped += 1
            continue

        abs_path, rel_path = _save_image_bytes(data, filename, dataset_id)
        try:
            w, h = get_image_dimensions(abs_path)
        except Exception:
            w, h = 0, 0

        phash = ""
        try:
            phash = calculate_phash(abs_path)
        except Exception:
            pass

        split_value = _split_from_yolo_path(zip_member)

        img_obj = Image(
            dataset_id=dataset_id,
            filename=Path(abs_path).name,
            filepath=rel_path,
            width=w, height=h,
            format=get_file_format(abs_path),
            file_hash=md5,
            phash=phash,
            split=split_value,
        )
        db.add(img_obj)
        pending.append((zip_member, img_obj))
        existing_hashes.add(md5)
        added += 1

        if len(pending) >= BATCH_SIZE:
            await db.flush()

    if pending or True:
        await db.flush()

    # Annotations
    ann_batch: list[Annotation] = []
    for zip_member, img_obj in pending:
        label_path = _find_label_file(zf, zip_member)
        if not label_path:
            continue
        try:
            label_text = zf.read(label_path).decode("utf-8")
        except Exception:
            continue

        for line in label_text.splitlines():
            parts_line = line.strip().split()
            if len(parts_line) < 5:
                continue
            try:
                yolo_idx = int(parts_line[0])
                cx, cy, bw, bh = (float(p) for p in parts_line[1:5])
            except ValueError:
                continue

            db_class_id = await resolve_class(yolo_idx)
            ann_batch.append(Annotation(
                image_id=img_obj.id,
                class_id=db_class_id,
                annotation_type="bbox",
                bbox_x=max(0.0, cx - bw / 2),
                bbox_y=max(0.0, cy - bh / 2),
                bbox_w=bw,
                bbox_h=bh,
            ))

            if len(ann_batch) >= BATCH_SIZE:
                db.add_all(ann_batch)
                await db.flush()
                ann_batch.clear()

    if ann_batch:
        db.add_all(ann_batch)
        await db.flush()

    return {"added": added, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def import_dataset_zip(
    db: AsyncSession,
    dataset_id: int,
    zip_bytes: bytes,
    existing_hashes: set[str],
    force_format: str | None = None,
) -> dict:
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        raise ValueError("Invalid ZIP file.")

    fmt = force_format or _detect_format(zf.namelist())

    if fmt == "coco":
        result = await _import_coco(db, dataset_id, zf, existing_hashes)
    elif fmt == "yolo":
        result = await _import_yolo(db, dataset_id, zf, existing_hashes)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return {"format": fmt, **result}
