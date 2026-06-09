"""
데이터셋 정제 라우터
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.sharding.router import get_sharded_db, get_meta_db
from app.models import Dataset
from app.services.refinement import (
    detect_duplicate_images,
    filter_annotations_by_bbox_size,
    delete_images_bulk,
    get_label_error_candidates,
)

router = APIRouter(prefix="/datasets/{dataset_id}/refinement", tags=["refinement"])


async def _require_dataset(dataset_id: int, meta_db: AsyncSession = Depends(get_meta_db)) -> Dataset:
    """메타 DB에서 데이터셋 존재 확인 (멀티샤드 안전)."""
    ds = await meta_db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다.")
    return ds


@router.get("/duplicates")
async def find_duplicates(
    dataset_id: int,
    db: AsyncSession = Depends(get_sharded_db),
    _ds: Dataset = Depends(_require_dataset),
):
    return await detect_duplicate_images(db, dataset_id)


class BboxFilterRequest(BaseModel):
    min_area: float = 0.0
    max_area: float = 1.0
    dry_run: bool = True


@router.post("/filter-bbox")
async def filter_bbox(
    dataset_id: int,
    req: BboxFilterRequest,
    db: AsyncSession = Depends(get_sharded_db),
    _ds: Dataset = Depends(_require_dataset),
):
    return await filter_annotations_by_bbox_size(
        db, dataset_id, req.min_area, req.max_area, req.dry_run
    )


class DeleteImagesRequest(BaseModel):
    image_ids: list[int]


@router.post("/delete-images")
async def delete_images(
    dataset_id: int,
    req: DeleteImagesRequest,
    db: AsyncSession = Depends(get_sharded_db),
):
    deleted = await delete_images_bulk(db, req.image_ids)
    return {"deleted": deleted}


class ResolveDuplicateRequest(BaseModel):
    keep_image_id: int | None = None
    delete_image_ids: list[int]


@router.post("/resolve-duplicate")
async def resolve_duplicate(
    dataset_id: int,
    req: ResolveDuplicateRequest,
    db: AsyncSession = Depends(get_sharded_db),
    _ds: Dataset = Depends(_require_dataset),
):
    if not req.delete_image_ids:
        return {"deleted": 0, "kept_image_id": req.keep_image_id}
    deleted = await delete_images_bulk(db, req.delete_image_ids)
    return {"deleted": deleted, "kept_image_id": req.keep_image_id}


@router.get("/label-errors")
async def label_errors(
    dataset_id: int,
    db: AsyncSession = Depends(get_sharded_db),
    _ds: Dataset = Depends(_require_dataset),
):
    return await get_label_error_candidates(db, dataset_id)
