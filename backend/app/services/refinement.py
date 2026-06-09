"""
데이터셋 정제 서비스 — 중복 탐지, 필터링, 레이블 오류 탐지
"""
import math
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from app.models import Image, Annotation, Class


async def detect_duplicate_images(db: AsyncSession, dataset_id: int) -> list[dict]:
    """퍼셉추얼 해시 기반 중복/유사 이미지 그룹 탐지"""
    stmt = select(Image).where(
        Image.dataset_id == dataset_id,
        Image.phash.isnot(None),
        Image.phash != "",
    )
    result = await db.execute(stmt)
    images = result.scalars().all()

    def _img_meta(img: Image) -> dict:
        """이미지 DB 행 → 모달에서 사용할 메타데이터 dict"""
        return {
            "id": img.id,
            "filename": img.filename,
            "phash": img.phash,
            "file_hash": img.file_hash,
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "created_at": img.created_at.isoformat() if img.created_at else None,
        }

    # 정확히 동일한 phash 그룹핑
    hash_groups: dict[str, list] = {}
    for img in images:
        hash_groups.setdefault(img.phash, []).append(_img_meta(img))

    duplicates = [
        {"phash": h, "images": group, "count": len(group)}
        for h, group in hash_groups.items()
        if len(group) > 1
    ]

    # MD5로 완전 중복 확인 (phash 그룹에 아직 없는 이미지도 포함하므로 전체 스캔)
    all_imgs_stmt = select(Image).where(
        Image.dataset_id == dataset_id,
        Image.file_hash.isnot(None),
    )
    all_imgs_result = await db.execute(all_imgs_stmt)
    all_imgs = all_imgs_result.scalars().all()

    md5_groups: dict[str, list] = {}
    for img in all_imgs:
        md5_groups.setdefault(img.file_hash, []).append(_img_meta(img))

    exact_duplicates = [
        {"file_hash": h, "images": group, "count": len(group)}
        for h, group in md5_groups.items()
        if len(group) > 1
    ]

    return {"perceptual": duplicates, "exact": exact_duplicates}


async def filter_annotations_by_bbox_size(
    db: AsyncSession,
    dataset_id: int,
    min_area: float,
    max_area: float,
    dry_run: bool = True,
) -> dict:
    """
    바운딩 박스 면적 범위로 주석 필터링.
    dry_run=True: 삭제될 건수만 반환
    dry_run=False: 실제 삭제 수행
    """
    stmt = (
        select(Annotation)
        .join(Image, Annotation.image_id == Image.id)
        .where(
            Image.dataset_id == dataset_id,
            Annotation.annotation_type == "bbox",
            Annotation.bbox_w.isnot(None),
            Annotation.bbox_h.isnot(None),
        )
    )
    result = await db.execute(stmt)
    annotations = result.scalars().all()

    to_delete = []
    for ann in annotations:
        area = (ann.bbox_w or 0) * (ann.bbox_h or 0)
        if area < min_area or area > max_area:
            to_delete.append(ann.id)

    if not dry_run and to_delete:
        await db.execute(
            delete(Annotation).where(Annotation.id.in_(to_delete))
        )
        await db.commit()

    return {
        "total_annotations": len(annotations),
        "to_delete": len(to_delete),
        "dry_run": dry_run,
    }


async def delete_images_bulk(
    db: AsyncSession, image_ids: list[int]
) -> int:
    """이미지 일괄 삭제 (cascade로 annotation도 삭제)"""
    from sqlalchemy import delete as sa_delete
    await db.execute(sa_delete(Image).where(Image.id.in_(image_ids)))
    await db.commit()
    return len(image_ids)


def _percentile(sorted_values: list[float], p: float) -> float:
    """정렬된 리스트에서 p번째 백분위수 반환 (선형 보간)"""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    rank = (p / 100) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


async def get_label_error_candidates(
    db: AsyncSession, dataset_id: int
) -> dict:
    """
    레이블 오류 후보 탐지:
    1) 클래스 미할당 주석
    2) bbox 면적 하위 1% 미만 (너무 작음)
    3) bbox 면적 상위 99% 초과 (너무 큼)
    4) bbox 좌표가 이미지 경계를 벗어남 (0~1 범위 초과)

    최소 20개 이상의 bbox가 있을 때만 백분위 이상치 탐지 수행.
    """
    errors: list[dict] = []

    # ── 1) 클래스 미할당 ──────────────────────────────────────────
    unassigned_stmt = (
        select(Annotation, Image)
        .join(Image, Annotation.image_id == Image.id)
        .where(Image.dataset_id == dataset_id, Annotation.class_id.is_(None))
    )
    unassigned_result = await db.execute(unassigned_stmt)
    for ann, img in unassigned_result:
        errors.append({
            "annotation_id": ann.id,
            "image_id": img.id,
            "image_filename": img.filename,
            "issue": "class_unassigned",
            "confidence": 1.0,
            "detail": None,
        })

    # ── 2+3+4) bbox 이상치 탐지 ───────────────────────────────────
    bbox_stmt = (
        select(Annotation, Image)
        .join(Image, Annotation.image_id == Image.id)
        .where(
            Image.dataset_id == dataset_id,
            Annotation.annotation_type == "bbox",
            Annotation.bbox_w.isnot(None),
            Annotation.bbox_h.isnot(None),
        )
    )
    bbox_result = await db.execute(bbox_stmt)
    bbox_rows = bbox_result.all()

    if bbox_rows:
        # 면적 계산 (정규화 좌표 기준, 0~1 범위)
        triples: list[tuple] = []
        for ann, img in bbox_rows:
            w = ann.bbox_w or 0.0
            h = ann.bbox_h or 0.0
            area = w * h
            triples.append((ann, img, area))

        # ── 2) 경계 초과 탐지 (좌표가 [0,1] 범위를 벗어난 경우) ──
        for ann, img, _ in triples:
            x = ann.bbox_x or 0.0
            y = ann.bbox_y or 0.0
            w = ann.bbox_w or 0.0
            h = ann.bbox_h or 0.0
            out_of_bounds = (
                x < 0 or y < 0
                or x + w > 1.001  # 소수점 오차 허용
                or y + h > 1.001
            )
            if out_of_bounds:
                errors.append({
                    "annotation_id": ann.id,
                    "image_id": img.id,
                    "image_filename": img.filename,
                    "issue": "bbox_out_of_bounds",
                    "confidence": 1.0,
                    "detail": {
                        "x": round(x, 4), "y": round(y, 4),
                        "w": round(w, 4), "h": round(h, 4),
                    },
                })

        # ── 3) 면적 백분위 이상치 (최소 20개 이상일 때만 적용) ───
        MIN_SAMPLES = 20
        if len(triples) >= MIN_SAMPLES:
            sorted_areas = sorted(a for _, _, a in triples)
            p1  = _percentile(sorted_areas, 1)
            p99 = _percentile(sorted_areas, 99)

            for ann, img, area in triples:
                if area == 0:
                    # 너비 또는 높이가 0인 주석
                    errors.append({
                        "annotation_id": ann.id,
                        "image_id": img.id,
                        "image_filename": img.filename,
                        "issue": "bbox_zero_area",
                        "confidence": 1.0,
                        "detail": {
                            "area": 0,
                            "w": round(ann.bbox_w or 0, 6),
                            "h": round(ann.bbox_h or 0, 6),
                        },
                    })
                elif area < p1:
                    # 하위 1% — 너무 작은 박스
                    conf = round(1.0 - (area / p1), 4) if p1 > 0 else 1.0
                    errors.append({
                        "annotation_id": ann.id,
                        "image_id": img.id,
                        "image_filename": img.filename,
                        "issue": "bbox_too_small",
                        "confidence": conf,
                        "detail": {
                            "area": round(area, 6),
                            "p1_threshold": round(p1, 6),
                        },
                    })
                elif area > p99:
                    # 상위 99% — 너무 큰 박스
                    conf = round(min((area / p99) - 1.0, 1.0), 4) if p99 > 0 else 1.0
                    errors.append({
                        "annotation_id": ann.id,
                        "image_id": img.id,
                        "image_filename": img.filename,
                        "issue": "bbox_too_large",
                        "confidence": conf,
                        "detail": {
                            "area": round(area, 6),
                            "p99_threshold": round(p99, 6),
                        },
                    })

    # ── 요약 통계 함께 반환 ───────────────────────────────────────
    issue_counts: dict[str, int] = {}
    for e in errors:
        issue_counts[e["issue"]] = issue_counts.get(e["issue"], 0) + 1

    return {
        "errors": errors,
        "total_errors": len(errors),
        "total_bbox_annotations": len(bbox_rows) if bbox_rows else 0,
        "issue_summary": issue_counts,
        "percentile_detection_applied": len(bbox_rows) >= 20 if bbox_rows else False,
    }
