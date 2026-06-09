"""클래스 관리 라우터"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.sharding.router import get_sharded_db
from app.models import Class
from app.schemas.class_ import ClassCreate, ClassRead, ClassUpdate
from app.services.class_service import get_or_create_class

router = APIRouter(prefix="/datasets/{dataset_id}/classes", tags=["classes"])

@router.get("", response_model=list[ClassRead])
async def list_classes(dataset_id: int, db: AsyncSession = Depends(get_sharded_db)):
    result = await db.execute(select(Class).where(Class.dataset_id == dataset_id).order_by(Class.id))
    return [ClassRead.model_validate(c) for c in result.scalars()]

@router.post("", response_model=ClassRead, status_code=status.HTTP_201_CREATED)
async def create_class(dataset_id: int, payload: ClassCreate, db: AsyncSession = Depends(get_sharded_db)):
    cls = await get_or_create_class(db, dataset_id, payload.name)
    if payload.color:
        cls.color = payload.color
    await db.commit()
    return ClassRead.model_validate(cls)

@router.patch("/{class_id}", response_model=ClassRead)
async def update_class(dataset_id: int, class_id: int, payload: ClassUpdate, db: AsyncSession = Depends(get_sharded_db)):
    cls = await db.get(Class, class_id)
    if not cls or cls.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="클래스를 찾을 수 없습니다.")
    for field, val in payload.model_dump(exclude_none=True).items():
        setattr(cls, field, val)
    await db.commit()
    await db.refresh(cls)
    return ClassRead.model_validate(cls)

@router.delete("/{class_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_class(dataset_id: int, class_id: int, db: AsyncSession = Depends(get_sharded_db)):
    cls = await db.get(Class, class_id)
    if not cls or cls.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="클래스를 찾을 수 없습니다.")
    await db.delete(cls)
    await db.commit()
