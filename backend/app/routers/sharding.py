"""
샤딩 관리 API
- GET  /api/v1/sharding/stats       샤드 분포 현황
- GET  /api/v1/sharding/dataset/{id} 특정 데이터셋의 샤드 정보
- POST /api/v1/sharding/rebalance   샤드 재배분 (관리자용)
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from app.sharding.router import shard_router, get_meta_db
from app.models import Dataset

router = APIRouter(prefix="/sharding", tags=["sharding"])


@router.get("/stats")
async def shard_stats():
    """샤드별 데이터셋 배분 현황과 연결 정보 반환."""
    stats = shard_router.stats()
    return {
        "shard_count": stats["shard_count"],
        "partition_count": stats["partition_count"],
        "distribution": {
            f"shard_{k}": v
            for k, v in stats["distribution"].items()
        },
        "total_datasets": sum(stats["distribution"].values()),
    }


@router.get("/dataset/{dataset_id}")
async def dataset_shard_info(dataset_id: int):
    """특정 데이터셋이 어느 샤드에 있는지 반환."""
    if shard_router.registry is None:
        raise HTTPException(status_code=503, detail="샤드 라우터 초기화 중")

    shard_id = await shard_router.registry.lookup(dataset_id)
    if shard_id is None:
        raise HTTPException(status_code=404, detail="샤드 배정 정보가 없습니다.")

    return {
        "dataset_id": dataset_id,
        "shard_id": shard_id,
        "partition": dataset_id % 8,  # HASH 파티션 번호
    }


@router.get("/health")
async def shard_health():
    """모든 샤드 DB 연결 상태 확인."""
    results = {}
    for shard_id, engine in shard_router._engines.items():
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            results[f"shard_{shard_id}"] = "healthy"
        except Exception as e:
            results[f"shard_{shard_id}"] = f"error: {e}"
    return results
