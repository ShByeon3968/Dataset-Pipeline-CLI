import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.database import create_tables
from app.services.file_handler import ensure_dirs
from app.sharding.router import shard_router

from app.routers import (
    datasets as datasets_router,
    images as images_router,
    annotations as annotations_router,
    classes as classes_router,
    analysis as analysis_router,
    refinement as refinement_router,
    ontology as ontology_router,
    export as export_router,
    augmentation as augmentation_router,
)
from app.routers.ontology import rules_router
from app.routers.analysis import analysis_router as coco_analysis_router
from app.routers import sharding as sharding_router
from app.routers.versions import router as versions_router
from app.routers.lineage import model_router as model_versions_router, lineage_router
from app.routers.auto_label import router as auto_label_router
from app.routers.onnx_models import router as onnx_models_router

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    await create_tables()
    await shard_router.initialize()

    # Reset stuck auto-label runs
    from app.models.auto_label_run import AutoLabelRun
    from sqlalchemy import update
    try:
        async with shard_router.get_meta_session() as session:
            async with session.begin():
                await session.execute(
                    update(AutoLabelRun)
                    .where(AutoLabelRun.status.in_(["pending", "running"]))
                    .values(status="failed", error_message="서버가 재시작되었거나 작업이 중단되었습니다.")
                )
    except Exception as e:
        logger.error("Error resetting stuck auto-label runs: %s", e)

    yield
    await shard_router.close()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="ML engineer one-stop dataset build/analyze/refine API",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(413)
async def request_entity_too_large(_: Request, __: Exception):
    return JSONResponse(
        status_code=413,
        content={"detail": "File too large. Max upload size is 500MB."},
    )


API_PREFIX = "/api/v1"
app.include_router(datasets_router.router, prefix=API_PREFIX)
app.include_router(images_router.router, prefix=API_PREFIX)
app.include_router(annotations_router.router, prefix=API_PREFIX)
app.include_router(classes_router.router, prefix=API_PREFIX)
app.include_router(analysis_router.router, prefix=API_PREFIX)
app.include_router(coco_analysis_router, prefix=API_PREFIX)
app.include_router(refinement_router.router, prefix=API_PREFIX)
app.include_router(ontology_router.router, prefix=API_PREFIX)
app.include_router(rules_router, prefix=API_PREFIX)
app.include_router(export_router.router, prefix=API_PREFIX)
app.include_router(sharding_router.router, prefix=API_PREFIX)
app.include_router(versions_router, prefix=API_PREFIX)
app.include_router(model_versions_router, prefix=API_PREFIX)
app.include_router(lineage_router, prefix=API_PREFIX)
app.include_router(auto_label_router, prefix=API_PREFIX)
app.include_router(onnx_models_router, prefix=API_PREFIX)
app.include_router(augmentation_router.router, prefix=API_PREFIX)


@app.get("/api/health")
async def health():
    from sqlalchemy import text
    from app.database import engine as meta_engine
    try:
        async with meta_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {
        "status": "ok" if db_ok else "degraded",
        "version": settings.app_version,
        "db": "ok" if db_ok else "unreachable",
    }
