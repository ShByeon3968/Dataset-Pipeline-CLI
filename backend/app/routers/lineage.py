"""
리니지 라우터
/api/v1/lineage/*          — 모델 버전 CRUD
/api/v1/datasets/{id}/lineage — 데이터셋 기준 리니지 그래프
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.sharding.router import get_meta_db
from app.schemas.version import (
    ModelVersionCreate, ModelVersionRead, ModelVersionList,
    ModelDatasetLinkCreate, ModelDatasetLinkRead,
    LineageGraph,
)
import app.services.versioning_service as svc

# ── 모델 버전 라우터 ──────────────────────────────────────────
model_router = APIRouter(prefix="/model-versions", tags=["lineage"])


@model_router.post("", response_model=ModelVersionRead, status_code=status.HTTP_201_CREATED)
async def create_model_version(
    payload: ModelVersionCreate,
    meta_db: AsyncSession = Depends(get_meta_db),
):
    return await svc.create_model_version(meta_db, payload)


@model_router.get("", response_model=ModelVersionList)
async def list_model_versions(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    meta_db: AsyncSession = Depends(get_meta_db),
):
    items, total = await svc.list_model_versions(meta_db, skip, limit)
    return ModelVersionList(items=items, total=total)


@model_router.get("/{model_version_id}", response_model=ModelVersionRead)
async def get_model_version(
    model_version_id: int,
    meta_db: AsyncSession = Depends(get_meta_db),
):
    mv = await svc.get_model_version(meta_db, model_version_id)
    if mv is None:
        raise HTTPException(status_code=404, detail="모델 버전을 찾을 수 없습니다.")
    return mv


@model_router.delete("/{model_version_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_version(
    model_version_id: int,
    meta_db: AsyncSession = Depends(get_meta_db),
):
    ok = await svc.delete_model_version(meta_db, model_version_id)
    if not ok:
        raise HTTPException(status_code=404, detail="모델 버전을 찾을 수 없습니다.")


# ── 모델↔데이터셋 버전 링크 ──────────────────────────────────
@model_router.post(
    "/{model_version_id}/links",
    response_model=ModelDatasetLinkRead,
    status_code=status.HTTP_201_CREATED,
)
async def link_dataset_version(
    model_version_id: int,
    payload: ModelDatasetLinkCreate,
    meta_db: AsyncSession = Depends(get_meta_db),
):
    mv = await svc.get_model_version(meta_db, model_version_id)
    if mv is None:
        raise HTTPException(status_code=404, detail="모델 버전을 찾을 수 없습니다.")
    return await svc.link_model_to_dataset_version(meta_db, model_version_id, payload)


@model_router.get("/{model_version_id}/links", response_model=list[ModelDatasetLinkRead])
async def list_links(
    model_version_id: int,
    meta_db: AsyncSession = Depends(get_meta_db),
):
    return await svc.list_model_links(meta_db, model_version_id)


@model_router.delete("/links/{link_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unlink_dataset_version(
    link_id: int,
    meta_db: AsyncSession = Depends(get_meta_db),
):
    ok = await svc.unlink_model_from_dataset_version(meta_db, link_id)
    if not ok:
        raise HTTPException(status_code=404, detail="링크를 찾을 수 없습니다.")


# ── 데이터셋 기준 리니지 그래프 ──────────────────────────────
lineage_router = APIRouter(tags=["lineage"])


@lineage_router.get(
    "/datasets/{dataset_id}/lineage",
    response_model=LineageGraph,
)
async def get_dataset_lineage(
    dataset_id: int,
    meta_db: AsyncSession = Depends(get_meta_db),
):
    """데이터셋에 연결된 모든 버전 + 모델 리니지 그래프 반환."""
    return await svc.build_lineage_graph(meta_db, dataset_id)
