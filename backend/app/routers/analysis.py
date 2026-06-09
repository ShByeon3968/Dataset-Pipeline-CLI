"""
데이터셋 분석 라우터

엔드포인트 목록
---------------
GET  /datasets/{id}/analysis/summary
GET  /datasets/{id}/analysis/class-distribution
GET  /datasets/{id}/analysis/bbox-stats
GET  /datasets/{id}/analysis/split-stats

GET  /datasets/{id}/analysis/embeddings?method=pca|umap|tsne
GET  /datasets/{id}/analysis/embeddings/outliers?top_k=20&duplicate_threshold=0.05
POST /datasets/{id}/analysis/embeddings/compute
GET  /datasets/{id}/analysis/embeddings/compute/status

POST /analysis/coco-json  (dataset 불필요)
"""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
import json

from app.sharding.router import get_sharded_db, get_meta_db
from app.models import Dataset
from app.services.analysis import (
    get_class_distribution, get_bbox_stats,
    get_dataset_summary, parse_coco_json,
    get_image_embeddings_2d, get_embedding_outliers,
)

router = APIRouter(prefix="/datasets/{dataset_id}/analysis", tags=["analysis"])


# -- 공통: 데이터셋 존재 확인 --

async def _require_dataset(
    dataset_id: int,
    meta_db: AsyncSession = Depends(get_meta_db),
) -> Dataset:
    ds = await meta_db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds


# -- 기본 통계 --

@router.get("/summary")
async def dataset_summary(
    dataset_id: int,
    db: AsyncSession = Depends(get_sharded_db),
    _ds: Dataset = Depends(_require_dataset),
):
    return await get_dataset_summary(db, dataset_id)


@router.get("/class-distribution")
async def class_distribution(
    dataset_id: int,
    db: AsyncSession = Depends(get_sharded_db),
    _ds: Dataset = Depends(_require_dataset),
):
    return await get_class_distribution(db, dataset_id)


@router.get("/bbox-stats")
async def bbox_stats(
    dataset_id: int,
    db: AsyncSession = Depends(get_sharded_db),
    _ds: Dataset = Depends(_require_dataset),
):
    return await get_bbox_stats(db, dataset_id)


@router.get("/split-stats")
async def split_stats(
    dataset_id: int,
    db: AsyncSession = Depends(get_sharded_db),
    _ds: Dataset = Depends(_require_dataset),
):
    """이미지 수를 split(train/val/test/unsplit) 별로 반환."""
    from sqlalchemy import select, func
    from app.models import Image

    result = await db.execute(
        select(Image.split, func.count(Image.id).label("count"))
        .where(Image.dataset_id == dataset_id)
        .group_by(Image.split)
    )
    rows = result.all()
    counts = {"train": 0, "val": 0, "test": 0, "unsplit": 0}
    total = 0
    for split_val, cnt in rows:
        key = split_val if split_val in ("train", "val", "test") else "unsplit"
        counts[key] += cnt
        total += cnt
    return {**counts, "total": total}


# -- 임베딩 시각화 --

@router.get("/embeddings")
async def image_embeddings(
    dataset_id: int,
    method: str = Query(
        "pca",
        description="투영 방식: pca | umap | tsne",
        pattern="^(pca|umap|tsne)$",
    ),
    db: AsyncSession = Depends(get_sharded_db),
    _ds: Dataset = Depends(_require_dataset),
):
    """
    이미지 픽셀 임베딩(CLIP/히스토그램)을 2D로 투영한 scatter plot 데이터.

    - method=pca  : numpy PCA (빠름, 선형)
    - method=umap : UMAP (느리지만 클러스터가 잘 드러남, umap-learn 필요)
    - method=tsne : t-SNE (중간 속도, scikit-learn)

    각 포인트에 thumbnail_url 이 포함되어 있어 프론트에서 툴팁 이미지로 사용 가능.
    임베딩 캐시가 없는 이미지는 요청 시점에 실시간 계산됩니다.
    (데이터셋이 크면 POST /embeddings/compute 를 먼저 호출하세요)
    """
    return await get_image_embeddings_2d(db, dataset_id, method=method)


@router.get("/embeddings/outliers")
async def embedding_outliers(
    dataset_id: int,
    top_k: int = Query(20, ge=1, le=200, description="이상치 상위 N개"),
    duplicate_threshold: float = Query(
        0.05, ge=0.0, le=1.0,
        description="중복 판정 코사인 거리 임계값 (낮을수록 엄격)",
    ),
    db: AsyncSession = Depends(get_sharded_db),
    _ds: Dataset = Depends(_require_dataset),
):
    """
    KNN 코사인 거리 기반 이상치 & 중복 후보 탐지.

    outliers            : 가장 가까운 이웃과 거리가 먼 이미지 top_k 개
    duplicate_candidates: 코사인 거리 < duplicate_threshold 인 이미지 쌍
    """
    return await get_embedding_outliers(
        db, dataset_id,
        top_k=top_k,
        duplicate_threshold=duplicate_threshold,
    )


# -- 임베딩 사전 계산 (백그라운드) --

_embed_tasks: dict[int, dict] = {}


@router.post("/embeddings/compute")
async def compute_embeddings(
    dataset_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_sharded_db),
    _ds: Dataset = Depends(_require_dataset),
):
    """
    데이터셋의 모든 이미지 임베딩을 백그라운드에서 계산합니다.
    이미 캐시된 이미지는 건너뜁니다.

    진행 상태 확인: GET /datasets/{id}/analysis/embeddings/compute/status
    """
    if _embed_tasks.get(dataset_id, {}).get("status") == "running":
        return {"status": "already_running", "dataset_id": dataset_id}

    _embed_tasks[dataset_id] = {"status": "running", "result": None}

    async def _run():
        from app.services.embedding_service import batch_compute
        try:
            result = await batch_compute(db, dataset_id)
            _embed_tasks[dataset_id] = {"status": "done", "result": result}
        except Exception as exc:
            _embed_tasks[dataset_id] = {"status": "error", "error": str(exc)}

    background_tasks.add_task(_run)
    return {"status": "started", "dataset_id": dataset_id}


@router.get("/embeddings/compute/status")
async def compute_embeddings_status(
    dataset_id: int,
    _ds: Dataset = Depends(_require_dataset),
):
    """임베딩 사전 계산 태스크의 진행 상태 조회."""
    info = _embed_tasks.get(dataset_id)
    if info is None:
        return {"status": "not_started", "dataset_id": dataset_id}
    return {"dataset_id": dataset_id, **info}


# -- 독립형: COCO JSON 분석 --

analysis_router = APIRouter(prefix="/analysis", tags=["analysis"])


@analysis_router.post("/coco-json")
async def analyze_coco_json(file: UploadFile = File(...)):
    """COCO JSON 파일을 업로드하면 클래스 분포 등 요약 정보를 반환."""
    content = await file.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    return parse_coco_json(data)
