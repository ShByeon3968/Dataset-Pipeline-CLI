"""
데이터셋 분석 서비스

주요 함수
---------
get_class_distribution   : 클래스별 어노테이션 수
get_bbox_stats           : bbox 크기 통계
get_dataset_summary      : 이미지/어노테이션/클래스/미레이블 수
parse_coco_json          : COCO JSON 파일 요약 (업로드 없이)
get_image_embeddings_2d  : 픽셀 임베딩 -> PCA/UMAP 2D 투영 (scatter plot)
get_embedding_outliers   : KNN 거리 기반 이상치.중복 후보 탐지
"""
from __future__ import annotations

import logging
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models import Image, Annotation, Class

logger = logging.getLogger(__name__)


# -- 클래스 분포 --

async def get_class_distribution(db: AsyncSession, dataset_id: int) -> list[dict]:
    stmt = (
        select(Class.name, Class.color, func.count(Annotation.id).label("count"))
        .join(Annotation, Annotation.class_id == Class.id, isouter=True)
        .join(Image, Annotation.image_id == Image.id, isouter=True)
        .where(Class.dataset_id == dataset_id)
        .group_by(Class.id, Class.name, Class.color)
        .order_by(func.count(Annotation.id).desc())
    )
    result = await db.execute(stmt)
    return [{"name": row.name, "color": row.color, "count": row.count} for row in result]


# -- BBox 통계 --

async def get_bbox_stats(db: AsyncSession, dataset_id: int) -> dict:
    from sqlalchemy import func as F

    stmt = (
        select(
            F.count(Annotation.id).label("cnt"),
            F.min(Annotation.bbox_w).label("w_min"),
            F.max(Annotation.bbox_w).label("w_max"),
            F.avg(Annotation.bbox_w).label("w_avg"),
            F.stddev(Annotation.bbox_w).label("w_std"),
            F.percentile_cont(0.5).within_group(Annotation.bbox_w).label("w_med"),
            F.min(Annotation.bbox_h).label("h_min"),
            F.max(Annotation.bbox_h).label("h_max"),
            F.avg(Annotation.bbox_h).label("h_avg"),
            F.stddev(Annotation.bbox_h).label("h_std"),
            F.percentile_cont(0.5).within_group(Annotation.bbox_h).label("h_med"),
            F.min(Annotation.bbox_w * Annotation.bbox_h).label("a_min"),
            F.max(Annotation.bbox_w * Annotation.bbox_h).label("a_max"),
            F.avg(Annotation.bbox_w * Annotation.bbox_h).label("a_avg"),
            F.stddev(Annotation.bbox_w * Annotation.bbox_h).label("a_std"),
            F.percentile_cont(0.5).within_group(
                Annotation.bbox_w * Annotation.bbox_h
            ).label("a_med"),
        )
        .join(Image, Annotation.image_id == Image.id)
        .where(
            Image.dataset_id == dataset_id,
            Annotation.annotation_type == "bbox",
            Annotation.bbox_w.isnot(None),
            Annotation.bbox_h.isnot(None),
        )
    )
    row = (await db.execute(stmt)).one()

    if not row.cnt:
        return {"count": 0, "width_stats": {}, "height_stats": {}, "area_stats": {}}

    def _fmt(v):
        return round(float(v), 2) if v is not None else 0.0

    return {
        "count": row.cnt,
        "width_stats":  {"min": _fmt(row.w_min), "max": _fmt(row.w_max), "mean": _fmt(row.w_avg), "median": _fmt(row.w_med), "std": _fmt(row.w_std)},
        "height_stats": {"min": _fmt(row.h_min), "max": _fmt(row.h_max), "mean": _fmt(row.h_avg), "median": _fmt(row.h_med), "std": _fmt(row.h_std)},
        "area_stats":   {"min": _fmt(row.a_min), "max": _fmt(row.a_max), "mean": _fmt(row.a_avg), "median": _fmt(row.a_med), "std": _fmt(row.a_std)},
    }


# -- 데이터셋 요약 --

async def get_dataset_summary(db: AsyncSession, dataset_id: int) -> dict:
    image_count = await db.scalar(
        select(func.count()).select_from(Image).where(Image.dataset_id == dataset_id)
    ) or 0
    annotation_count = await db.scalar(
        select(func.count())
        .select_from(Annotation)
        .join(Image)
        .where(Image.dataset_id == dataset_id)
    ) or 0
    class_count = await db.scalar(
        select(func.count()).select_from(Class).where(Class.dataset_id == dataset_id)
    ) or 0
    unlabeled_count = await db.scalar(
        select(func.count())
        .select_from(Image)
        .outerjoin(Annotation, Annotation.image_id == Image.id)
        .where(Image.dataset_id == dataset_id, Annotation.id.is_(None))
    ) or 0
    return {
        "image_count": image_count,
        "annotation_count": annotation_count,
        "class_count": class_count,
        "unlabeled_count": unlabeled_count,
        "avg_annotations_per_image": (
            round(annotation_count / image_count, 2) if image_count > 0 else 0
        ),
    }


# -- COCO JSON 파일 분석 --

def parse_coco_json(coco_data: dict) -> dict:
    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])

    cat_map = {c["id"]: c["name"] for c in categories}
    class_counts: dict[str, int] = {}
    areas = []
    for ann in annotations:
        cat_name = cat_map.get(ann.get("category_id"), "Unknown")
        class_counts[cat_name] = class_counts.get(cat_name, 0) + 1
        bbox = ann.get("bbox")
        if bbox and len(bbox) == 4:
            areas.append(bbox[2] * bbox[3])

    return {
        "image_count": len(images),
        "annotation_count": len(annotations),
        "class_count": len(categories),
        "class_distribution": [
            {"name": k, "count": v} for k, v in class_counts.items()
        ],
        "area_stats": {
            "min": min(areas) if areas else 0,
            "max": max(areas) if areas else 0,
            "mean": round(sum(areas) / len(areas), 2) if areas else 0,
        },
    }


# -- 내부 헬퍼: 임베딩 행렬 수집 --

async def _collect_embeddings(
    db: AsyncSession,
    dataset_id: int,
) -> tuple[list[dict], np.ndarray]:
    """
    DB에서 이미지 목록을 읽고 임베딩 행렬을 구성.

    Returns:
        meta  : [{"image_id", "filename", "filepath", "class_names", "class_colors"}, ...]
        matrix: ndarray shape (N, D)
    """
    from app.services.embedding_service import get_or_compute

    img_rows = (
        await db.execute(
            select(Image.id, Image.filename, Image.filepath)
            .where(Image.dataset_id == dataset_id)
            .order_by(Image.id)
        )
    ).all()

    if not img_rows:
        return [], np.empty((0, 0), dtype=np.float32)

    # 이미지별 클래스 목록 조회
    ann_rows = (
        await db.execute(
            select(
                Annotation.image_id,
                Class.name.label("class_name"),
                Class.color.label("class_color"),
            )
            .join(Image, Annotation.image_id == Image.id)
            .outerjoin(Class, Annotation.class_id == Class.id)
            .where(Image.dataset_id == dataset_id)
            .distinct()
        )
    ).all()

    img_classes: dict[int, list[str]] = {}
    img_colors: dict[int, list[str]] = {}
    for row in ann_rows:
        if row.class_name:
            img_classes.setdefault(row.image_id, []).append(row.class_name)
            img_colors.setdefault(row.image_id, []).append(row.class_color or "#888888")

    meta: list[dict] = []
    vecs: list[np.ndarray] = []

    for img_id, filename, filepath in img_rows:
        vec = get_or_compute(img_id, filepath, dataset_id)
        if vec is None:
            continue
        meta.append({
            "image_id": img_id,
            "filename": filename,
            "filepath": filepath,
            "thumbnail_url": f"/api/v1/datasets/{dataset_id}/images/{img_id}/file",
            "class_names": img_classes.get(img_id, []),
            "class_colors": img_colors.get(img_id, []),
        })
        vecs.append(vec)

    if not vecs:
        return [], np.empty((0, 0), dtype=np.float32)

    matrix = np.vstack(vecs).astype(np.float32)
    return meta, matrix


# -- 2D 투영 --

def _project_pca(matrix: np.ndarray) -> np.ndarray:
    """numpy 전용 PCA -> (N, 2)."""
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    top2 = eigenvectors[:, idx[:2]]
    return (centered @ top2).astype(np.float32)


def _project_umap(matrix: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """UMAP 2D 투영. umap-learn 없으면 PCA 폴백."""
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, len(matrix) - 1),
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(matrix).astype(np.float32)
    except ImportError:
        logger.warning("umap-learn 미설치 -- PCA 폴백")
        return _project_pca(matrix)


def _project_tsne(matrix: np.ndarray, perplexity: int = 30) -> np.ndarray:
    """t-SNE 2D 투영 (scikit-learn)."""
    from sklearn.manifold import TSNE
    perp = min(perplexity, max(2, len(matrix) - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=500)
    return tsne.fit_transform(matrix).astype(np.float32)


# -- 공개 분석 함수 --

async def get_image_embeddings_2d(
    db: AsyncSession,
    dataset_id: int,
    method: str = "pca",
) -> dict:
    """
    이미지 픽셀 임베딩을 2D로 투영한 scatter plot 데이터 반환.

    Parameters
    ----------
    method : "pca" | "umap" | "tsne"

    Response shape
    --------------
    {
      "points": [
        {
          "image_id": 3,
          "filename": "cat.jpg",
          "x": 1.23, "y": -0.45,
          "thumbnail_url": "/api/v1/datasets/1/images/3/file",
          "class_names": ["cat"],
          "class_colors": ["#ff0000"],
          "label": "cat"
        }, ...
      ],
      "total": 42,
      "method": "umap",
      "embedding_model": "clip"
    }
    """
    meta, matrix = await _collect_embeddings(db, dataset_id)

    if len(meta) == 0:
        return {
            "points": [], "total": 0, "method": method,
            "embedding_model": "none",
            "note": "임베딩 가능한 이미지가 없습니다.",
        }

    if len(meta) < 2:
        return {
            "points": [], "total": len(meta), "method": method,
            "embedding_model": "unknown",
            "note": "투영에 최소 2개 이미지가 필요합니다.",
        }

    method = method.lower()
    if method == "umap":
        projected = _project_umap(matrix)
    elif method == "tsne":
        projected = _project_tsne(matrix)
    else:
        projected = _project_pca(matrix)
        method = "pca"

    # 임베딩 모델 이름 추론 (벡터 차원으로 구분: CLIP=512, histogram=192)
    dim = matrix.shape[1]
    emb_model = "clip" if dim == 512 else "histogram"

    points = []
    for i, m in enumerate(meta):
        label = m["class_names"][0] if m["class_names"] else "unlabeled"
        points.append({
            "image_id": m["image_id"],
            "filename": m["filename"],
            "x": round(float(projected[i, 0]), 4),
            "y": round(float(projected[i, 1]), 4),
            "thumbnail_url": m["thumbnail_url"],
            "class_names": m["class_names"],
            "class_colors": m["class_colors"],
            "label": label,
        })

    return {
        "points": points,
        "total": len(points),
        "method": method,
        "embedding_model": emb_model,
    }


async def get_embedding_outliers(
    db: AsyncSession,
    dataset_id: int,
    top_k: int = 20,
    duplicate_threshold: float = 0.05,
) -> dict:
    """
    KNN 거리 기반 이상치 & 중복 후보 탐지.

    이상치  : 가장 가까운 이웃과의 코사인 거리가 큰 이미지 (top_k 개)
    중복 후보: 가장 가까운 이웃과의 코사인 거리 < duplicate_threshold 인 쌍

    Response
    --------
    {
      "outliers": [
        {"image_id", "filename", "thumbnail_url", "nn_distance", "class_names"}
      ],
      "duplicate_candidates": [
        {"image_a": {...}, "image_b": {...}, "distance": 0.01}
      ],
      "total_images": 100
    }
    """
    meta, matrix = await _collect_embeddings(db, dataset_id)

    if len(meta) < 2:
        return {
            "outliers": [],
            "duplicate_candidates": [],
            "total_images": len(meta),
            "note": "분석에 최소 2개 이미지가 필요합니다.",
        }

    from sklearn.metrics.pairwise import cosine_distances

    dist_matrix = cosine_distances(matrix)   # (N, N)
    np.fill_diagonal(dist_matrix, np.inf)    # 자기 자신 제외

    nn_distances = dist_matrix.min(axis=1)   # 각 이미지의 최근접 이웃 거리
    nn_indices = dist_matrix.argmin(axis=1)  # 최근접 이웃 인덱스

    # 이상치: nn_distance 내림차순 top_k
    outlier_idx = np.argsort(nn_distances)[::-1][:top_k]
    outliers = []
    for idx in outlier_idx:
        m = meta[idx]
        outliers.append({
            "image_id": m["image_id"],
            "filename": m["filename"],
            "thumbnail_url": m["thumbnail_url"],
            "nn_distance": round(float(nn_distances[idx]), 4),
            "class_names": m["class_names"],
        })

    # 중복 후보: nn_distance < threshold (i < j 쌍만)
    dup_candidates = []
    seen: set[frozenset] = set()
    for i, (dist, j) in enumerate(zip(nn_distances, nn_indices)):
        if dist >= duplicate_threshold:
            continue
        pair: frozenset = frozenset([i, int(j)])
        if pair in seen:
            continue
        seen.add(pair)
        ma, mb = meta[i], meta[int(j)]
        dup_candidates.append({
            "image_a": {
                "image_id": ma["image_id"],
                "filename": ma["filename"],
                "thumbnail_url": ma["thumbnail_url"],
            },
            "image_b": {
                "image_id": mb["image_id"],
                "filename": mb["filename"],
                "thumbnail_url": mb["thumbnail_url"],
            },
            "distance": round(float(dist), 4),
        })

    # 거리 오름차순 정렬 (가장 비슷한 것 먼저)
    dup_candidates.sort(key=lambda x: x["distance"])

    return {
        "outliers": outliers,
        "duplicate_candidates": dup_candidates,
        "total_images": len(meta),
    }
