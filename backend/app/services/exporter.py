"""
Dataset export service — COCO JSON / YOLO / Pascal VOC

Export folder structure (all three formats):
  train/
    images/           <- image files
    _annotations.coco.json   (COCO only)
    labels/           <- *.txt files  (YOLO only)
    annotations/      <- *.xml files  (VOC only)
  val/
    ...
  test/
    ...
  data.yaml           (YOLO only — root level)
  build_manifest.json

Split assignment:
  - Images with Image.split set (imported from split directories):
      used as-is.
  - Images with Image.split = None (imported without split info):
      randomly assigned according to train_ratio / val_ratio / test_ratio.
      Assignment is deterministic: sorted by image id, then sliced by ratio.
"""
import json
import math
import os
import zipfile
from pathlib import Path
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models import Image, Annotation, Class, Dataset
from app.core.config import get_settings
from app.services.file_handler import get_file_bytes

settings = get_settings()

VALID_SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def _assign_splits(
    images: list,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list]:
    """
    Returns {"train": [...], "val": [...], "test": [...]} where every image
    appears in exactly one bucket.

    Images that already have a split value are placed directly.
    Images with split=None are distributed proportionally (sorted by id).
    """
    # Normalize ratios so they sum to 1
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
        total = 1.0
    tr = train_ratio / total
    vr = val_ratio / total
    # test_ratio derived so all three always sum to 1

    buckets: dict[str, list] = {"train": [], "val": [], "test": []}
    unsplit: list = []

    for img in images:
        if img.split in VALID_SPLITS:
            buckets[img.split].append(img)
        else:
            unsplit.append(img)

    if unsplit:
        # Sort by id for deterministic assignment
        unsplit.sort(key=lambda x: x.id)
        n = len(unsplit)
        n_train = math.ceil(n * tr)
        n_val = math.ceil(n * vr)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        buckets["train"].extend(unsplit[:n_train])
        buckets["val"].extend(unsplit[n_train:n_train + n_val])
        buckets["test"].extend(unsplit[n_train + n_val:])

    return buckets


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

async def _build_manifest(
    db: AsyncSession,
    dataset: Dataset,
    classes: list,
    split_buckets: dict[str, list],
    export_format: str,
) -> dict:
    """Build build_manifest.json dict with per-split statistics."""
    class_id_to_name = {cls.id: cls.name for cls in classes}
    all_images = [img for imgs in split_buckets.values() for img in imgs]
    all_image_ids = [img.id for img in all_images]

    # Annotation counts per (image_id, class_id)
    if all_image_ids:
        rows = (
            await db.execute(
                select(Annotation.image_id, Annotation.class_id, func.count(Annotation.id))
                .where(Annotation.image_id.in_(all_image_ids))
                .group_by(Annotation.image_id, Annotation.class_id)
            )
        ).all()
    else:
        rows = []

    # Build lookup: image_id -> {class_name: count}
    img_ann_counts: dict[int, dict[str, int]] = {}
    for img_id, class_id, cnt in rows:
        cname = class_id_to_name.get(class_id, "unknown")
        img_ann_counts.setdefault(img_id, {}).setdefault(cname, 0)
        img_ann_counts[img_id][cname] += cnt

    splits_section: dict = {}
    for split_name in VALID_SPLITS:
        imgs = split_buckets.get(split_name, [])
        if not imgs:
            continue
        ann_by_class: dict[str, int] = {}
        total_anns = 0
        for img in imgs:
            for cname, cnt in img_ann_counts.get(img.id, {}).items():
                ann_by_class[cname] = ann_by_class.get(cname, 0) + cnt
                total_anns += cnt
        splits_section[split_name] = {
            "num_images": len(imgs),
            "num_annotations": total_anns,
            "canonical_annotation_counts": ann_by_class,
        }

    return {
        "dataset_name": dataset.name,
        "export_format": export_format,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "canonical_classes": sorted(cls.name for cls in classes),
        "statistics": {
            "num_images": len(all_images),
            "num_annotations": sum(
                s["num_annotations"] for s in splits_section.values()
            ),
            "annotation_counts_by_class": {
                cname: sum(
                    splits_section[sp]["canonical_annotation_counts"].get(cname, 0)
                    for sp in splits_section
                )
                for cname in sorted(class_id_to_name.values())
            },
        },
        # source_datasets and sampling_policy are not tracked by this pipeline.
        "source_datasets": [],
        "sampling_policy": None,
        "splits": splits_section,
    }


# ---------------------------------------------------------------------------
# ZIP helper
# ---------------------------------------------------------------------------

def _make_zip(dataset_id: int, fmt: str, text_files: dict, split_buckets: dict) -> str:
    """Bundle text/JSON files and split-organized images into a ZIP."""
    os.makedirs(settings.exports_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(
        settings.exports_dir, f"dataset_{dataset_id}_{fmt}_{timestamp}.zip"
    )

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Text/JSON files (annotations, manifest, yaml, etc.)
        for name, content in text_files.items():
            zf.writestr(name, content)

        # Images organized by split
        for split_name, imgs in split_buckets.items():
            for img in imgs:
                if img.filepath:
                    file_bytes = get_file_bytes(img.filepath)
                    if file_bytes:
                        zf.writestr(f"{split_name}/images/{img.filename}", file_bytes)

    return zip_path


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

async def export_coco(
    db: AsyncSession,
    dataset_id: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> str:
    """Export as COCO JSON with per-split folder structure."""
    dataset = await db.get(Dataset, dataset_id)
    images = (
        await db.execute(select(Image).where(Image.dataset_id == dataset_id))
    ).scalars().all()
    classes = (
        await db.execute(select(Class).where(Class.dataset_id == dataset_id))
    ).scalars().all()

    split_buckets = _assign_splits(list(images), train_ratio, val_ratio, test_ratio)
    class_id_to_name = {cls.id: cls.name for cls in classes}

    text_files: dict[str, str] = {}

    for split_name, imgs in split_buckets.items():
        if not imgs:
            continue

        img_ids = [img.id for img in imgs]
        anns_result = (
            await db.execute(
                select(Annotation).where(Annotation.image_id.in_(img_ids))
            )
        ).scalars().all()
        anns_by_img: dict[int, list] = {}
        for ann in anns_result:
            anns_by_img.setdefault(ann.image_id, []).append(ann)

        coco = {
            "info": {
                "description": f"{dataset.name} — {split_name}",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat(),
            },
            "categories": [
                {"id": cls.id, "name": cls.name, "supercategory": "object"}
                for cls in classes
            ],
            "images": [],
            "annotations": [],
        }

        ann_id = 1
        for img in imgs:
            coco["images"].append({
                "id": img.id,
                "file_name": img.filename,
                "width": img.width or 0,
                "height": img.height or 0,
            })
            for ann in anns_by_img.get(img.id, []):
                if ann.bbox_w is None:
                    continue
                w_px = ann.bbox_w * (img.width or 1)
                h_px = ann.bbox_h * (img.height or 1)
                x_px = ann.bbox_x * (img.width or 1)
                y_px = ann.bbox_y * (img.height or 1)
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img.id,
                    "category_id": ann.class_id,
                    "bbox": [round(x_px, 2), round(y_px, 2), round(w_px, 2), round(h_px, 2)],
                    "area": round(w_px * h_px, 2),
                    "iscrowd": 0,
                })
                ann_id += 1

        text_files[f"{split_name}/_annotations.coco.json"] = json.dumps(
            coco, ensure_ascii=False, indent=2
        )

    manifest = await _build_manifest(db, dataset, list(classes), split_buckets, "coco")
    text_files["build_manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2)

    return _make_zip(dataset_id, "coco", text_files, split_buckets)


async def export_yolo(
    db: AsyncSession,
    dataset_id: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> str:
    """Export as YOLO with per-split folder structure and data.yaml."""
    dataset = await db.get(Dataset, dataset_id)
    classes = (
        await db.execute(select(Class).where(Class.dataset_id == dataset_id))
    ).scalars().all()
    images = (
        await db.execute(select(Image).where(Image.dataset_id == dataset_id))
    ).scalars().all()

    split_buckets = _assign_splits(list(images), train_ratio, val_ratio, test_ratio)
    class_list = list(classes)
    class_id_to_idx = {cls.id: i for i, cls in enumerate(class_list)}

    text_files: dict[str, str] = {}

    for split_name, imgs in split_buckets.items():
        if not imgs:
            continue

        img_ids = [img.id for img in imgs]
        anns_result = (
            await db.execute(
                select(Annotation).where(Annotation.image_id.in_(img_ids))
            )
        ).scalars().all()
        anns_by_img: dict[int, list] = {}
        for ann in anns_result:
            anns_by_img.setdefault(ann.image_id, []).append(ann)

        for img in imgs:
            lines = []
            for ann in anns_by_img.get(img.id, []):
                if ann.class_id is None or ann.bbox_w is None:
                    continue
                idx = class_id_to_idx.get(ann.class_id, 0)
                cx = ann.bbox_x + ann.bbox_w / 2
                cy = ann.bbox_y + ann.bbox_h / 2
                lines.append(f"{idx} {cx:.6f} {cy:.6f} {ann.bbox_w:.6f} {ann.bbox_h:.6f}")
            text_files[f"{split_name}/labels/{Path(img.filename).stem}.txt"] = "\n".join(lines)

    # data.yaml at root
    non_empty_splits = [sp for sp in VALID_SPLITS if split_buckets.get(sp)]
    data_yaml_lines = [
        "path: .",
        *[f"{sp}: {sp}/images" for sp in non_empty_splits],
        f"nc: {len(class_list)}",
        f"names: [{', '.join(cls.name for cls in class_list)}]",
    ]
    text_files["data.yaml"] = "\n".join(data_yaml_lines) + "\n"
    text_files["classes.txt"] = "\n".join(cls.name for cls in class_list)

    manifest = await _build_manifest(db, dataset, class_list, split_buckets, "yolo")
    text_files["build_manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2)

    return _make_zip(dataset_id, "yolo", text_files, split_buckets)


async def export_pascal_voc(
    db: AsyncSession,
    dataset_id: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> str:
    """Export as Pascal VOC XML with per-split folder structure."""
    try:
        from lxml import etree
    except ImportError:
        import xml.etree.ElementTree as etree

    dataset = await db.get(Dataset, dataset_id)
    classes_result = (
        await db.execute(select(Class).where(Class.dataset_id == dataset_id))
    ).scalars().all()
    class_id_to_name = {cls.id: cls.name for cls in classes_result}
    images = (
        await db.execute(select(Image).where(Image.dataset_id == dataset_id))
    ).scalars().all()

    split_buckets = _assign_splits(list(images), train_ratio, val_ratio, test_ratio)
    text_files: dict[str, str] = {}

    for split_name, imgs in split_buckets.items():
        if not imgs:
            continue

        img_ids = [img.id for img in imgs]
        anns_result = (
            await db.execute(
                select(Annotation).where(Annotation.image_id.in_(img_ids))
            )
        ).scalars().all()
        anns_by_img: dict[int, list] = {}
        for ann in anns_result:
            anns_by_img.setdefault(ann.image_id, []).append(ann)

        for img in imgs:
            root = etree.Element("annotation")
            etree.SubElement(root, "filename").text = img.filename
            size = etree.SubElement(root, "size")
            etree.SubElement(size, "width").text = str(img.width or 0)
            etree.SubElement(size, "height").text = str(img.height or 0)
            etree.SubElement(size, "depth").text = "3"

            for ann in anns_by_img.get(img.id, []):
                if ann.bbox_w is None:
                    continue
                w = img.width or 1
                h = img.height or 1
                obj = etree.SubElement(root, "object")
                etree.SubElement(obj, "name").text = class_id_to_name.get(ann.class_id, "Unknown")
                etree.SubElement(obj, "difficult").text = "0"
                bndbox = etree.SubElement(obj, "bndbox")
                etree.SubElement(bndbox, "xmin").text = str(int(ann.bbox_x * w))
                etree.SubElement(bndbox, "ymin").text = str(int(ann.bbox_y * h))
                etree.SubElement(bndbox, "xmax").text = str(int((ann.bbox_x + ann.bbox_w) * w))
                etree.SubElement(bndbox, "ymax").text = str(int((ann.bbox_y + ann.bbox_h) * h))

            xml_str = etree.tostring(root, pretty_print=True, encoding="unicode")
            text_files[f"{split_name}/annotations/{Path(img.filename).stem}.xml"] = xml_str

    manifest = await _build_manifest(
        db, dataset, list(classes_result), split_buckets, "pascal_voc"
    )
    text_files["build_manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2)

    return _make_zip(dataset_id, "voc", text_files, split_buckets)
