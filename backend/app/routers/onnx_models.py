"""
ONNX 모델 관리 라우터

POST /onnx-models/upload    — 파일 업로드 + 메타 저장
GET  /onnx-models           — 목록
GET  /onnx-models/{id}      — 단건 조회
GET  /onnx-models/{id}/validate — 텐서 shape 확인
DELETE /onnx-models/{id}    — 파일 + DB 삭제
"""
import json
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.sharding.router import get_meta_db
from app.models.onnx_model import OnnxModel
from app.core.config import get_settings

router = APIRouter(prefix="/onnx-models", tags=["onnx-models"])
logger = logging.getLogger(__name__)
settings = get_settings()


# ── 스키마 ────────────────────────────────────────────────────────────────

class OnnxModelRead(BaseModel):
    id: int
    name: str
    architecture: str
    class_labels: list[str]
    file_size: int | None
    input_width: int
    input_height: int
    conf_threshold: float
    iou_threshold: float

    model_config = {"from_attributes": True}


def _to_read(obj: OnnxModel) -> OnnxModelRead:
    return OnnxModelRead(
        id=obj.id,
        name=obj.name,
        architecture=obj.architecture,
        class_labels=json.loads(obj.class_labels),
        file_size=obj.file_size,
        input_width=obj.input_width,
        input_height=obj.input_height,
        conf_threshold=obj.conf_threshold,
        iou_threshold=obj.iou_threshold,
    )


# ── 엔드포인트 ────────────────────────────────────────────────────────────

@router.post("/upload", response_model=OnnxModelRead, status_code=201)
async def upload_onnx_model(
    file: UploadFile = File(...),
    name: str = Form(""),
    architecture: str = Form("yolov8"),
    class_labels: str = Form("[]"),
    input_width: int = Form(640),
    input_height: int = Form(640),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    db: AsyncSession = Depends(get_meta_db),
):
    if not (file.filename or "").endswith(".onnx"):
        raise HTTPException(status_code=400, detail="ONNX 파일만 업로드 가능합니다.")

    data = await file.read()
    fname = f"{uuid.uuid4().hex}.onnx"
    dest = Path(settings.models_dir) / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)

    try:
        labels = json.loads(class_labels)
    except json.JSONDecodeError:
        labels = []

    obj = OnnxModel(
        name=name or file.filename,
        architecture=architecture,
        file_path=fname,
        file_size=len(data),
        class_labels=json.dumps(labels),
        input_width=input_width,
        input_height=input_height,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return _to_read(obj)


@router.get("", response_model=list[OnnxModelRead])
async def list_onnx_models(db: AsyncSession = Depends(get_meta_db)):
    rows = (
        await db.execute(select(OnnxModel).order_by(OnnxModel.id.desc()))
    ).scalars().all()
    return [_to_read(r) for r in rows]


@router.get("/{model_id}", response_model=OnnxModelRead)
async def get_onnx_model(model_id: int, db: AsyncSession = Depends(get_meta_db)):
    obj = await db.get(OnnxModel, model_id)
    if not obj:
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다.")
    return _to_read(obj)


@router.get("/{model_id}/validate")
async def validate_onnx_model(model_id: int, db: AsyncSession = Depends(get_meta_db)):
    obj = await db.get(OnnxModel, model_id)
    if not obj:
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다.")
    abs_path = str(Path(settings.models_dir) / obj.file_path)
    from app.services.onnx_inference import validate_model
    return validate_model(abs_path)


@router.delete("/{model_id}", status_code=204)
async def delete_onnx_model(model_id: int, db: AsyncSession = Depends(get_meta_db)):
    obj = await db.get(OnnxModel, model_id)
    if not obj:
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다.")
    p = Path(settings.models_dir) / obj.file_path
    if p.exists():
        p.unlink()
    from app.services.onnx_inference import evict_session
    evict_session(model_id)
    await db.delete(obj)
    await db.commit()
