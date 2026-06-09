from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database import get_db
from app.models import Dataset, Image, Annotation, Class
from app.schemas.dataset import DatasetCreate, DatasetUpdate, DatasetRead, DatasetList
from app.sharding.router import shard_router

router = APIRouter(prefix="/datasets", tags=["datasets"])


async def _fetch_counts(db: AsyncSession, dataset_id: int) -> tuple[int, int, int]:
    """Return (image_count, annotation_count, class_count) for a dataset."""
    img_count = await db.scalar(
        select(func.count()).select_from(Image).where(Image.dataset_id == dataset_id)
    ) or 0
    ann_count = await db.scalar(
        select(func.count())
        .select_from(Annotation)
        .join(Image, Annotation.image_id == Image.id)
        .where(Image.dataset_id == dataset_id)
    ) or 0
    cls_count = await db.scalar(
        select(func.count()).select_from(Class).where(Class.dataset_id == dataset_id)
    ) or 0
    return img_count, ann_count, cls_count


@router.get("", response_model=DatasetList)
async def list_datasets(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Dataset).order_by(Dataset.created_at.desc()))
    datasets = result.scalars().all()

    # Single-pass bulk count query per dataset to avoid N*3 round trips
    dataset_ids = [ds.id for ds in datasets]
    if not dataset_ids:
        return DatasetList(items=[], total=0)

    img_counts = {
        row[0]: row[1]
        for row in (await db.execute(
            select(Image.dataset_id, func.count(Image.id))
            .where(Image.dataset_id.in_(dataset_ids))
            .group_by(Image.dataset_id)
        )).all()
    }
    ann_counts = {
        row[0]: row[1]
        for row in (await db.execute(
            select(Image.dataset_id, func.count(Annotation.id))
            .join(Image, Annotation.image_id == Image.id)
            .where(Image.dataset_id.in_(dataset_ids))
            .group_by(Image.dataset_id)
        )).all()
    }
    cls_counts = {
        row[0]: row[1]
        for row in (await db.execute(
            select(Class.dataset_id, func.count(Class.id))
            .where(Class.dataset_id.in_(dataset_ids))
            .group_by(Class.dataset_id)
        )).all()
    }

    items = [
        DatasetRead(
            id=ds.id, name=ds.name, description=ds.description,
            source=ds.source, created_at=ds.created_at, updated_at=ds.updated_at,
            image_count=img_counts.get(ds.id, 0),
            annotation_count=ann_counts.get(ds.id, 0),
            class_count=cls_counts.get(ds.id, 0),
        )
        for ds in datasets
    ]
    return DatasetList(items=items, total=len(items))


@router.post("", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def create_dataset(payload: DatasetCreate, db: AsyncSession = Depends(get_db)):
    ds = Dataset(**payload.model_dump())
    db.add(ds)
    await db.commit()
    await db.refresh(ds)
    shard_id = await shard_router.assign_dataset(ds.id)
    return DatasetRead(
        id=ds.id, name=ds.name, description=ds.description,
        source=ds.source, created_at=ds.created_at, updated_at=ds.updated_at,
        image_count=0, annotation_count=0, class_count=0,
        shard_id=shard_id,
    )


@router.get("/{dataset_id}", response_model=DatasetRead)
async def get_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    img_count, ann_count, cls_count = await _fetch_counts(db, dataset_id)
    return DatasetRead(
        id=ds.id, name=ds.name, description=ds.description,
        source=ds.source, created_at=ds.created_at, updated_at=ds.updated_at,
        image_count=img_count, annotation_count=ann_count, class_count=cls_count,
    )


@router.patch("/{dataset_id}", response_model=DatasetRead)
async def update_dataset(
    dataset_id: int, payload: DatasetUpdate, db: AsyncSession = Depends(get_db)
):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    for field, val in payload.model_dump(exclude_none=True).items():
        setattr(ds, field, val)
    await db.commit()
    await db.refresh(ds)
    img_count, ann_count, cls_count = await _fetch_counts(db, dataset_id)
    return DatasetRead(
        id=ds.id, name=ds.name, description=ds.description,
        source=ds.source, created_at=ds.created_at, updated_at=ds.updated_at,
        image_count=img_count, annotation_count=ann_count, class_count=cls_count,
    )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    await db.delete(ds)
    await shard_router.remove_dataset(dataset_id)
