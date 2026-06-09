import uuid
from typing import Dict, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.sharding.router import get_sharded_db, shard_router
from app.database import get_db
from app.models import Dataset
from app.routers.images import _task_store

router = APIRouter(prefix="/datasets/{dataset_id}/augment", tags=["augmentation"])

class AugmentRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.8
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def start_augmentation(
    dataset_id: int,
    req: AugmentRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Start a background task to augment images using Qwen-Edit Multiple-angles model.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다.")

    task_id = str(uuid.uuid4())
    _task_store[task_id] = {"status": "pending", "result": None, "error": None}

    async def _run():
        _task_store[task_id]["status"] = "running"
        bg_session = await shard_router.get_session_for_dataset(dataset_id)
        try:
            from app.services.augmentation_service import run_augmentation_task
            result = await run_augmentation_task(
                bg_session,
                dataset_id,
                req.prompt,
                req.negative_prompt,
                req.strength,
                req.num_inference_steps,
                req.guidance_scale
            )
            _task_store[task_id] = {"status": "done", "result": result, "error": None}
        except Exception as e:
            _task_store[task_id] = {"status": "error", "result": None, "error": str(e)}
        finally:
            await bg_session.close()

    background_tasks.add_task(_run)
    return {"task_id": task_id, "status": "pending"}
