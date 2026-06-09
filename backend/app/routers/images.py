"""
이미지 업로드 및 관리 라우터

filepath 컬럼에는 상대경로가 저장되므로,
실제 파일 접근 시 resolve_filepath()로 절대경로를 복원합니다.
"""
import io as _io
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, status, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel as _BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete as sa_delete

from app.sharding.router import get_sharded_db
from app.models import Image, Annotation, Dataset
from app.schemas.image import ImageRead, ImageList
from app.services.file_handler import (
    save_uploaded_file, extract_zip_and_get_images,
    calculate_md5_bytes, calculate_md5, calculate_phash,
    get_image_dimensions, get_file_format, resolve_filepath,
    delete_file, SUPPORTED_FORMATS,
)

router = APIRouter(prefix="/datasets/{dataset_id}/images", tags=["images"])

# ── 백그라운드 태스크 상태 저장소 (프로세스 재시작 시 초기화) ────────
# 영속성이 필요한 경우 Redis hash로 교체하세요.
_task_store: dict[str, dict] = {}


# ── 태스크 상태 조회 ─────────────────────────────────────────────────

@router.get("/tasks/{task_id}")
async def get_import_task_status(task_id: str):
    """백그라운드 임포트 태스크 진행 상태 조회"""
    info = _task_store.get(task_id)
    if not info:
        raise HTTPException(status_code=404, detail="태스크를 찾을 수 없습니다.")
    return info


# ── 이미지 목록 ──────────────────────────────────────────────────────

@router.get("", response_model=ImageList)
async def list_images(
    dataset_id: int,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_sharded_db),
):
    total = await db.scalar(
        select(func.count()).select_from(Image).where(Image.dataset_id == dataset_id)
    ) or 0
    result = await db.execute(
        select(Image)
        .where(Image.dataset_id == dataset_id)
        .order_by(Image.id)
        .offset(skip)
        .limit(limit)
    )
    images = result.scalars().all()
    return ImageList(
        items=[ImageRead.model_validate(img) for img in images],
        total=total,
    )


# ── 개별 파일 업로드 ─────────────────────────────────────────────────

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_images(
    dataset_id: int,
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_sharded_db),
):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다.")

    existing = await db.execute(
        select(Image.file_hash).where(Image.dataset_id == dataset_id)
    )
    existing_hashes = {row[0] for row in existing if row[0]}

    added, skipped, errors = [], [], []
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            errors.append({"filename": upload.filename, "reason": "지원하지 않는 형식"})
            continue
        data = await upload.read()
        md5 = calculate_md5_bytes(data)
        if md5 in existing_hashes:
            skipped.append(upload.filename)
            continue
        try:
            rel_path = save_uploaded_file(data, upload.filename or "upload", dataset_id)
            abs_path = resolve_filepath(rel_path)
            w, h = get_image_dimensions(abs_path)
            phash = calculate_phash(abs_path)
            fmt = get_file_format(abs_path)
            img = Image(
                dataset_id=dataset_id,
                filename=Path(abs_path).name,
                filepath=rel_path,
                width=w, height=h, format=fmt,
                file_hash=md5, phash=phash,
            )
            db.add(img)
            existing_hashes.add(md5)
            added.append(upload.filename)
        except Exception as e:
            errors.append({"filename": upload.filename, "reason": str(e)})

    await db.commit()
    return {"added": len(added), "skipped": len(skipped), "errors": errors}


# ── ZIP 업로드 (어노테이션 없음) ─────────────────────────────────────

@router.post("/upload-zip", status_code=status.HTTP_201_CREATED)
async def upload_zip(
    dataset_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_sharded_db),
):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다.")

    data = await file.read()
    existing = await db.execute(
        select(Image.file_hash).where(Image.dataset_id == dataset_id)
    )
    existing_hashes = {row[0] for row in existing if row[0]}

    rel_paths = extract_zip_and_get_images(data, dataset_id)
    added = 0
    for rel_path in rel_paths:
        abs_path = resolve_filepath(rel_path)
        md5 = calculate_md5(abs_path)
        if md5 in existing_hashes:
            continue
        w, h = get_image_dimensions(abs_path)
        phash = calculate_phash(abs_path)
        fmt = get_file_format(abs_path)
        img = Image(
            dataset_id=dataset_id,
            filename=Path(abs_path).name,
            filepath=rel_path,
            width=w, height=h, format=fmt,
            file_hash=md5, phash=phash,
        )
        db.add(img)
        existing_hashes.add(md5)
        added += 1

    await db.commit()
    return {"added": added}


# ── ZIP + 어노테이션 업로드 (백그라운드 처리) ────────────────────────

@router.post("/upload-zip-annotated", status_code=status.HTTP_202_ACCEPTED)
async def upload_zip_annotated(
    dataset_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fmt: str | None = Query(default=None, description="coco | yolo | None(자동감지)"),
    db: AsyncSession = Depends(get_sharded_db),
):
    """
    COCO / YOLO 어노테이션 포함 ZIP 업로드.
    즉시 task_id를 반환하고, 실제 임포트는 백그라운드에서 처리합니다.
    GET /datasets/{dataset_id}/images/tasks/{task_id} 로 진행 상태를 확인하세요.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다.")

    if fmt and fmt.lower() not in ("coco", "yolo"):
        raise HTTPException(status_code=400, detail="fmt는 'coco' 또는 'yolo'만 허용됩니다.")

    zip_bytes = await file.read()

    existing = await db.execute(
        select(Image.file_hash).where(Image.dataset_id == dataset_id)
    )
    existing_hashes = {row[0] for row in existing if row[0]}

    task_id = str(uuid.uuid4())
    _task_store[task_id] = {"status": "pending", "result": None, "error": None}

    from app.services.import_service import import_dataset_zip
    from app.sharding.router import shard_router

    async def _run():
        _task_store[task_id]["status"] = "running"
        # 백그라운드에서는 새로운 DB 세션을 사용
        bg_session = await shard_router.get_session_for_dataset(dataset_id)
        try:
            result = await import_dataset_zip(
                bg_session, dataset_id, zip_bytes, existing_hashes,
                force_format=fmt.lower() if fmt else None,
            )
            await bg_session.commit()
            _task_store[task_id] = {"status": "done", "result": result, "error": None}
        except Exception as e:
            await bg_session.rollback()
            _task_store[task_id] = {"status": "error", "result": None, "error": str(e)}
        finally:
            await bg_session.close()

    background_tasks.add_task(_run)
    return {"task_id": task_id, "status": "pending"}


# ── 단일 이미지 조회 ─────────────────────────────────────────────────

@router.get("/{image_id}", response_model=ImageRead)
async def get_image(
    dataset_id: int, image_id: int, db: AsyncSession = Depends(get_sharded_db)
):
    img = await db.get(Image, image_id)
    if not img or img.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
    return ImageRead.model_validate(img)


# ── 이미지 파일 서빙 ─────────────────────────────────────────────────

@router.get("/{image_id}/file")
async def serve_image_file(
    dataset_id: int, image_id: int, db: AsyncSession = Depends(get_sharded_db)
):
    img = await db.get(Image, image_id)
    if not img or img.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")

    abs_path = resolve_filepath(img.filepath)
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    suffix = Path(abs_path).suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".bmp": "image/bmp",
        ".gif": "image/gif", ".tiff": "image/tiff",
        ".tif": "image/tiff", ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "application/octet-stream")
    return FileResponse(abs_path, media_type=media_type)


# ── 이미지 삭제 ──────────────────────────────────────────────────────

@router.delete("/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_image(
    dataset_id: int, image_id: int, db: AsyncSession = Depends(get_sharded_db)
):
    img = await db.get(Image, image_id)
    if not img or img.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")

    await db.execute(sa_delete(Annotation).where(Annotation.image_id == image_id))

    try:
        abs_path = resolve_filepath(img.filepath)
        delete_file(abs_path)
    except Exception:
        pass  # 파일이 없어도 DB 레코드는 삭제

    await db.delete(img)
    await db.commit()


# ── Roboflow 가져오기 (백그라운드 처리) ──────────────────────────────

class RoboflowImportRequest(_BaseModel):
    api_key: str
    workspace: str
    project_id: str
    version: int = 1


@router.post("/import-roboflow", status_code=status.HTTP_202_ACCEPTED)
async def import_from_roboflow(
    dataset_id: int,
    req: RoboflowImportRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_sharded_db),
):
    """
    Roboflow 프로젝트를 COCO 형식으로 다운로드 후 백그라운드에서 가져오기.
    즉시 task_id를 반환합니다.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다.")

    try:
        from app.services.roboflow_client import download_dataset
    except ImportError:
        raise HTTPException(status_code=501, detail="roboflow 패키지가 설치되지 않았습니다.")

    from app.core.config import get_settings as _settings
    from app.sharding.router import shard_router
    from app.services.import_service import import_dataset_zip

    settings = _settings()
    dest = os.path.join(
        settings.uploads_dir,
        f"roboflow_{dataset_id}_{req.project_id}_v{req.version}",
    )

    existing = await db.execute(
        select(Image.file_hash).where(Image.dataset_id == dataset_id)
    )
    existing_hashes = {row[0] for row in existing if row[0]}

    task_id = str(uuid.uuid4())
    _task_store[task_id] = {"status": "pending", "result": None, "error": None}

    req_copy = req.model_copy()  # BackgroundTasks에 안전하게 전달

    async def _run():
        _task_store[task_id]["status"] = "running"
        bg_session = await shard_router.get_session_for_dataset(dataset_id)
        try:
            download_dir = download_dataset(
                req_copy.api_key, req_copy.workspace,
                req_copy.project_id, req_copy.version, dest,
            )
            # 다운로드 디렉토리를 메모리 ZIP으로 묶어 import_service 처리
            import zipfile as _zf
            buf = _io.BytesIO()
            with _zf.ZipFile(buf, "w", _zf.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(download_dir):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        arcname = os.path.relpath(fpath, download_dir)
                        zf.write(fpath, arcname)
            zip_bytes = buf.getvalue()

            result = await import_dataset_zip(
                bg_session, dataset_id, zip_bytes, existing_hashes,
                force_format="coco",
            )
            await bg_session.commit()
            _task_store[task_id] = {"status": "done", "result": result, "error": None}
        except Exception as e:
            await bg_session.rollback()
            _task_store[task_id] = {"status": "error", "result": None, "error": str(e)}
        finally:
            await bg_session.close()

    background_tasks.add_task(_run)
    return {"task_id": task_id, "status": "pending"}
