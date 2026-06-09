"""
버저닝 라우터
/api/v1/datasets/{dataset_id}/versions/*
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.sharding.router import get_sharded_db, get_meta_db
from app.schemas.version import (
    DatasetVersionCreate, DatasetVersionRead, DatasetVersionList,
)
import app.services.versioning_service as svc

router = APIRouter(
    prefix="/datasets/{dataset_id}/versions",
    tags=["versions"],
)


@router.post("", response_model=DatasetVersionRead, status_code=status.HTTP_201_CREATED)
async def create_version(
    dataset_id: int,
    payload: DatasetVersionCreate,
    meta_db: AsyncSession = Depends(get_meta_db),
    shard_db: AsyncSession = Depends(get_sharded_db),
):
    """현재 데이터셋 상태 스냅샷을 찍어 새 버전으로 저장."""
    return await svc.create_dataset_version(meta_db, shard_db, dataset_id, payload)


@router.get("", response_model=DatasetVersionList)
async def list_versions(
    dataset_id: int,
    branch: str | None = Query(None, description="브랜치 필터"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    meta_db: AsyncSession = Depends(get_meta_db),
):
    items, total = await svc.list_dataset_versions(meta_db, dataset_id, branch, skip, limit)
    return DatasetVersionList(items=items, total=total)


@router.get("/{version_id}", response_model=DatasetVersionRead)
async def get_version(
    dataset_id: int,
    version_id: int,
    meta_db: AsyncSession = Depends(get_meta_db),
):
    ver = await svc.get_dataset_version(meta_db, version_id)
    if ver is None or ver.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="버전을 찾을 수 없습니다.")
    return ver


@router.delete("/{version_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_version(
    dataset_id: int,
    version_id: int,
    meta_db: AsyncSession = Depends(get_meta_db),
):
    ver = await svc.get_dataset_version(meta_db, version_id)
    if ver is None or ver.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="버전을 찾을 수 없습니다.")
    await svc.delete_dataset_version(meta_db, version_id)
