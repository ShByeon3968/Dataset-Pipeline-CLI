"""
주석(레이블) CRUD 라우터
URL: /datasets/{dataset_id}/images/{image_id}/annotations

dataset_id 를 경로에 포함하여 올바른 샤드 세션을 주입받습니다.
image가 해당 dataset에 속하는지 검증도 함께 수행합니다.
"""
import json
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.sharding.router import get_sharded_db
from app.models import Annotation, Image, Class
from app.schemas.annotation import AnnotationCreate, AnnotationUpdate, AnnotationRead

router = APIRouter(
    prefix="/datasets/{dataset_id}/images/{image_id}/annotations",
    tags=["annotations"],
)


def _enrich(ann: Annotation, cls: Class | None) -> AnnotationRead:
    seg = json.loads(ann.segmentation) if ann.segmentation else []
    return AnnotationRead(
        id=ann.id, image_id=ann.image_id,
        class_id=ann.class_id,
        class_name=cls.name if cls else None,
        class_color=cls.color if cls else None,
        bbox_x=ann.bbox_x, bbox_y=ann.bbox_y,
        bbox_w=ann.bbox_w, bbox_h=ann.bbox_h,
        segmentation=seg,
        annotation_type=ann.annotation_type,
        is_auto_generated=ann.is_auto_generated,
        confidence=ann.confidence,
        created_at=ann.created_at, updated_at=ann.updated_at,
    )


async def _verify_image(db: AsyncSession, dataset_id: int, image_id: int) -> Image:
    """이미지가 존재하고 해당 dataset에 속하는지 확인"""
    img = await db.get(Image, image_id)
    if not img or img.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
    return img


@router.get("", response_model=list[AnnotationRead])
async def list_annotations(
    dataset_id: int,
    image_id: int,
    db: AsyncSession = Depends(get_sharded_db),
):
    await _verify_image(db, dataset_id, image_id)
    result = await db.execute(
        select(Annotation, Class)
        .outerjoin(Class, Annotation.class_id == Class.id)
        .where(Annotation.image_id == image_id)
        .order_by(Annotation.id)
    )
    return [_enrich(ann, cls) for ann, cls in result]


@router.post("", response_model=AnnotationRead, status_code=status.HTTP_201_CREATED)
async def create_annotation(
    dataset_id: int,
    image_id: int,
    payload: AnnotationCreate,
    db: AsyncSession = Depends(get_sharded_db),
):
    await _verify_image(db, dataset_id, image_id)

    ann = Annotation(
        image_id=image_id,
        class_id=payload.class_id,
        bbox_x=payload.bbox_x, bbox_y=payload.bbox_y,
        bbox_w=payload.bbox_w, bbox_h=payload.bbox_h,
        annotation_type=payload.annotation_type,
        segmentation=json.dumps(payload.segmentation) if payload.segmentation else None,
    )
    db.add(ann)
    await db.commit()
    await db.refresh(ann)

    cls = await db.get(Class, ann.class_id) if ann.class_id else None
    return _enrich(ann, cls)


@router.put("/{annotation_id}", response_model=AnnotationRead)
async def update_annotation(
    dataset_id: int,
    image_id: int,
    annotation_id: int,
    payload: AnnotationUpdate,
    db: AsyncSession = Depends(get_sharded_db),
):
    await _verify_image(db, dataset_id, image_id)
    ann = await db.get(Annotation, annotation_id)
    if not ann or ann.image_id != image_id:
        raise HTTPException(status_code=404, detail="주석을 찾을 수 없습니다.")

    for field, val in payload.model_dump(exclude_none=True).items():
        setattr(ann, field, val)

    await db.commit()
    await db.refresh(ann)
    cls = await db.get(Class, ann.class_id) if ann.class_id else None
    return _enrich(ann, cls)


@router.delete("/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_annotation(
    dataset_id: int,
    image_id: int,
    annotation_id: int,
    db: AsyncSession = Depends(get_sharded_db),
):
    await _verify_image(db, dataset_id, image_id)
    ann = await db.get(Annotation, annotation_id)
    if not ann or ann.image_id != image_id:
        raise HTTPException(status_code=404, detail="주석을 찾을 수 없습니다.")

    await db.delete(ann)
    await db.commit()
