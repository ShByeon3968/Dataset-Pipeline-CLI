import json
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func

from app.database import get_db
from app.models import Dataset, Image, Annotation, Class
from app.models.auto_label_run import AutoLabelRun
from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auto-label", tags=["auto-label"])
settings = get_settings()


# ── 스키마 ────────────────────────────────────────────────────────────────

class AutoLabelRequest(BaseModel):
    mode: Literal["yolo_world", "onnx"] = "yolo_world"
    # YOLO-World 전용
    text_prompts: list[str] = Field(default=[], description="yolo_world 모드에서 필수")
    # ONNX 전용
    onnx_model_id: int | None = Field(default=None, description="onnx 모드에서 필수")
    # 공통
    confidence_threshold: float = Field(default=0.25, ge=0.01, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    overwrite: bool = Field(default=False)


class AutoLabelRunRead(BaseModel):
    id: int
    dataset_id: int
    model_name: str
    confidence_threshold: float
    iou_threshold: float
    text_prompts: str | None = None
    onnx_model_id: int | None = None
    status: str
    total_images: int
    processed_images: int
    total_annotations: int
    error_message: str | None = None

    model_config = {"from_attributes": True}


class AutoLabelRunList(BaseModel):
    items: list[AutoLabelRunRead]
    total: int


# ── 백그라운드 태스크 ──────────────────────────────────────────────────────

async def _run_auto_label(
    run_id: int,
    dataset_id: int,
    mode: str,
    text_prompts: list[str],
    onnx_model_id: int | None,
    confidence_threshold: float,
    iou_threshold: float,
    overwrite: bool,
):
    from app.services.file_handler import resolve_filepath
    from app.sharding.router import shard_router

    meta_session = shard_router.get_meta_session()
    shard_session = await shard_router.get_session_for_dataset(dataset_id)

    # ONNX 모드: 모델 메타 미리 로드
    onnx_meta = None
    onnx_labels: list[str] = []
    if mode == "onnx" and onnx_model_id is not None:
        async with shard_router.get_meta_session() as ms:
            from app.models.onnx_model import OnnxModel
            onnx_meta = await ms.get(OnnxModel, onnx_model_id)
            if not onnx_meta:
                logger.error("ONNX 모델 %d 를 찾을 수 없음", onnx_model_id)
                async with shard_router.get_meta_session() as ms2:
                    await ms2.execute(
                        update(AutoLabelRun)
                        .where(AutoLabelRun.id == run_id)
                        .values(status="failed", error_message=f"ONNX 모델 {onnx_model_id} 없음")
                    )
                    await ms2.commit()
                return
            onnx_labels = json.loads(onnx_meta.class_labels)

    try:
        # pending → running
        async with meta_session:
            db_run = await meta_session.get(AutoLabelRun, run_id)
            if not db_run or db_run.status != "pending":
                return
            db_run.status = "running"
            await meta_session.commit()

        async with shard_session:
            img_result = await shard_session.execute(
                select(Image).where(Image.dataset_id == dataset_id)
            )
            images = img_result.scalars().all()

            cls_result = await shard_session.execute(
                select(Class).where(Class.dataset_id == dataset_id)
            )
            class_map = {c.name: c for c in cls_result.scalars().all()}

            if overwrite:
                ids_result = await shard_session.execute(
                    select(Annotation.id)
                    .join(Image, Annotation.image_id == Image.id)
                    .where(
                        Image.dataset_id == dataset_id,
                        Annotation.is_auto_generated == True,
                    )
                )
                ids_to_delete = [r[0] for r in ids_result.all()]
                if ids_to_delete:
                    await shard_session.execute(
                        delete(Annotation).where(Annotation.id.in_(ids_to_delete))
                    )
                await shard_session.commit()

            total = len(images)
            processed = 0
            total_anns = 0

            for img in images:
                # 취소 확인
                async with shard_router.get_meta_session() as ms_check:
                    db_run = await ms_check.get(AutoLabelRun, run_id)
                    if db_run and db_run.status != "running":
                        return

                try:
                    abs_path = resolve_filepath(img.filepath)

                    # ── 모드별 추론 분기 ──────────────────────────────
                    if mode == "onnx" and onnx_meta is not None:
                        from app.services.onnx_inference import run_inference
                        detections = run_inference(
                            model_id=onnx_model_id,
                            file_path=onnx_meta.file_path,
                            models_dir=settings.models_dir,
                            architecture=onnx_meta.architecture,
                            class_labels=onnx_labels,
                            image_abs_path=abs_path,
                            input_w=onnx_meta.input_width,
                            input_h=onnx_meta.input_height,
                            conf_threshold=confidence_threshold,
                            iou_threshold=iou_threshold,
                        )
                    else:
                        from app.services.auto_label_service import predict_image
                        detections = predict_image(abs_path, text_prompts, confidence_threshold)
                    # ─────────────────────────────────────────────────

                    for det in detections:
                        cname = det["class_name"]
                        if cname not in class_map:
                            from app.services.class_service import get_or_create_class
                            new_cls = await get_or_create_class(shard_session, dataset_id, cname)
                            class_map[cname] = new_cls

                        seg_json = None
                        if det.get("segmentation"):
                            seg_json = json.dumps(det["segmentation"])

                        ann = Annotation(
                            image_id=img.id,
                            class_id=class_map[cname].id,
                            bbox_x=det["bbox"]["x"],
                            bbox_y=det["bbox"]["y"],
                            bbox_w=det["bbox"]["w"],
                            bbox_h=det["bbox"]["h"],
                            segmentation=seg_json,
                            annotation_type="bbox",
                            is_auto_generated=True,
                            confidence=det["confidence"],
                            source_prompt=json.dumps(text_prompts) if mode == "yolo_world" else None,
                            auto_label_run_id=run_id,
                        )
                        shard_session.add(ann)
                        total_anns += 1

                    await shard_session.commit()

                except FileNotFoundError:
                    logger.warning("이미지 파일 없음: %s", img.filepath)
                except Exception as e:
                    logger.error("이미지 %s 처리 오류: %s", img.id, e)
                finally:
                    processed += 1

                if processed % 10 == 0 or processed == total:
                    async with shard_router.get_meta_session() as ms:
                        await ms.execute(
                            update(AutoLabelRun)
                            .where(AutoLabelRun.id == run_id)
                            .values(processed_images=processed, total_annotations=total_anns)
                        )
                        await ms.commit()

        async with shard_router.get_meta_session() as ms2:
            await ms2.execute(
                update(AutoLabelRun)
                .where(AutoLabelRun.id == run_id)
                .values(status="completed", processed_images=total, total_annotations=total_anns)
            )
            await ms2.commit()

    except Exception as e:
        logger.error("Auto-label run %s 실패: %s", run_id, e)
        async with shard_router.get_meta_session() as ms3:
            await ms3.execute(
                update(AutoLabelRun)
                .where(AutoLabelRun.id == run_id)
                .values(status="failed", error_message=str(e)[:1000])
            )
            await ms3.commit()


# ── 엔드포인트 ────────────────────────────────────────────────────────────

@router.post("/datasets/{dataset_id}/runs", response_model=AutoLabelRunRead, status_code=202)
async def start_auto_label(
    dataset_id: int,
    req: AutoLabelRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    ds = await db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if req.mode == "yolo_world" and not req.text_prompts:
        raise HTTPException(status_code=422, detail="yolo_world 모드에서는 text_prompts가 필요합니다.")
    if req.mode == "onnx" and req.onnx_model_id is None:
        raise HTTPException(status_code=422, detail="onnx 모드에서는 onnx_model_id가 필요합니다.")

    running = await db.scalar(
        select(AutoLabelRun).where(
            AutoLabelRun.dataset_id == dataset_id,
            AutoLabelRun.status.in_(["pending", "running"]),
        )
    )
    if running:
        raise HTTPException(status_code=409, detail="이미 실행 중인 작업이 있습니다.")

    total_images = await db.scalar(
        select(func.count(Image.id)).where(Image.dataset_id == dataset_id)
    ) or 0

    if req.mode == "onnx":
        model_name = f"onnx:{req.onnx_model_id}"
    else:
        model_name = "yolo-world"

    run = AutoLabelRun(
        dataset_id=dataset_id,
        model_name=model_name,
        confidence_threshold=req.confidence_threshold,
        iou_threshold=req.iou_threshold,
        text_prompts=json.dumps(req.text_prompts) if req.mode == "yolo_world" else None,
        onnx_model_id=req.onnx_model_id,
        status="pending",
        total_images=total_images,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    background_tasks.add_task(
        _run_auto_label,
        run_id=run.id,
        dataset_id=dataset_id,
        mode=req.mode,
        text_prompts=req.text_prompts,
        onnx_model_id=req.onnx_model_id,
        confidence_threshold=req.confidence_threshold,
        iou_threshold=req.iou_threshold,
        overwrite=req.overwrite,
    )
    return run


@router.get("/datasets/{dataset_id}/runs", response_model=AutoLabelRunList)
async def list_runs(dataset_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(AutoLabelRun)
        .where(AutoLabelRun.dataset_id == dataset_id)
        .order_by(AutoLabelRun.created_at.desc())
    )
    runs = result.scalars().all()
    return AutoLabelRunList(items=list(runs), total=len(runs))


@router.get("/datasets/{dataset_id}/runs/{run_id}", response_model=AutoLabelRunRead)
async def get_run(dataset_id: int, run_id: int, db: AsyncSession = Depends(get_db)):
    run = await db.scalar(
        select(AutoLabelRun).where(
            AutoLabelRun.id == run_id,
            AutoLabelRun.dataset_id == dataset_id,
        )
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.delete("/datasets/{dataset_id}/runs/{run_id}/annotations", status_code=204)
async def delete_auto_annotations(dataset_id: int, run_id: int):
    from app.sharding.router import shard_router
    shard_session = await shard_router.get_session_for_dataset(dataset_id)
    async with shard_session:
        ids_result = await shard_session.execute(
            select(Annotation.id)
            .join(Image, Annotation.image_id == Image.id)
            .where(
                Image.dataset_id == dataset_id,
                Annotation.auto_label_run_id == run_id,
            )
        )
        ids = [r[0] for r in ids_result.all()]
        if ids:
            await shard_session.execute(delete(Annotation).where(Annotation.id.in_(ids)))
            await shard_session.commit()


@router.post("/datasets/{dataset_id}/runs/{run_id}/cancel", response_model=AutoLabelRunRead)
async def cancel_auto_label(
    dataset_id: int,
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    run = await db.scalar(
        select(AutoLabelRun).where(
            AutoLabelRun.id == run_id,
            AutoLabelRun.dataset_id == dataset_id,
        )
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status not in ["pending", "running"]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel a run with status '{run.status}'")

    run.status = "failed"
    run.error_message = "Stopped by user"
    await db.commit()
    await db.refresh(run)
    return run
