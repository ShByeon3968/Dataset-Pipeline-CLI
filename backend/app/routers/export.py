"""
Dataset export router
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.sharding.router import get_sharded_db
from app.models import Dataset
from app.services.exporter import export_coco, export_yolo, export_pascal_voc
import os

router = APIRouter(prefix="/datasets/{dataset_id}/export", tags=["export"])

FORMAT_MAP = {
    "coco": export_coco,
    "yolo": export_yolo,
    "voc": export_pascal_voc,
}


@router.get("/{format}")
async def export_dataset(
    dataset_id: int,
    format: str,
    train_ratio: float = Query(default=0.7, ge=0.0, le=1.0, description="Train split ratio (used for images without a split label)"),
    val_ratio: float = Query(default=0.2, ge=0.0, le=1.0, description="Val split ratio"),
    test_ratio: float = Query(default=0.1, ge=0.0, le=1.0, description="Test split ratio (remainder after train+val if omitted)"),
    db: AsyncSession = Depends(get_sharded_db),
):
    """
    Export dataset as ZIP in the specified format.

    - **format**: coco | yolo | voc
    - **train_ratio / val_ratio / test_ratio**: only applied to images that were
      imported without a split label (case 2 & 3). Images already tagged
      train/val/test retain their original split.
    - Ratios are auto-normalized, so they don't need to sum to exactly 1.

    ZIP structure:
    ```
    train/images/...
    train/_annotations.coco.json  (COCO)
    train/labels/*.txt            (YOLO)
    train/annotations/*.xml       (VOC)
    val/...
    test/...
    data.yaml                     (YOLO only)
    build_manifest.json
    ```
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    fmt = format.lower()
    exporter = FORMAT_MAP.get(fmt)
    if not exporter:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {format}. Available: coco, yolo, voc",
        )

    try:
        zip_path = await exporter(
            db,
            dataset_id,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")

    if not os.path.exists(zip_path):
        raise HTTPException(status_code=500, detail="File creation failed.")

    filename = os.path.basename(zip_path)
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
