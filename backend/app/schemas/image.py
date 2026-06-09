from datetime import datetime
from pydantic import BaseModel
from typing import Optional
from .annotation import AnnotationRead


class ImageCreate(BaseModel):
    dataset_id: int
    filename: str
    filepath: str
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    file_hash: Optional[str] = None
    phash: Optional[str] = None


class ImageRead(BaseModel):
    id: int
    dataset_id: int
    filename: str
    filepath: str
    width: Optional[int]
    height: Optional[int]
    format: Optional[str]
    file_hash: Optional[str]
    phash: Optional[str]
    created_at: datetime
    annotations: list[AnnotationRead] = []

    model_config = {"from_attributes": True}


class ImageList(BaseModel):
    items: list[ImageRead]
    total: int
