from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class AnnotationCreate(BaseModel):
    image_id: int
    class_id: Optional[int] = None
    bbox_x: Optional[float] = None
    bbox_y: Optional[float] = None
    bbox_w: Optional[float] = None
    bbox_h: Optional[float] = None
    segmentation: Optional[list] = None
    annotation_type: str = "bbox"


class AnnotationUpdate(BaseModel):
    class_id: Optional[int] = None
    bbox_x: Optional[float] = None
    bbox_y: Optional[float] = None
    bbox_w: Optional[float] = None
    bbox_h: Optional[float] = None
    annotation_type: Optional[str] = None


class AnnotationRead(BaseModel):
    id: int
    image_id: int
    class_id: Optional[int]
    class_name: Optional[str] = None
    class_color: Optional[str] = None
    bbox_x: Optional[float]
    bbox_y: Optional[float]
    bbox_w: Optional[float]
    bbox_h: Optional[float]
    segmentation: Optional[list] = None
    annotation_type: str
    is_auto_generated: bool = False
    confidence: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
