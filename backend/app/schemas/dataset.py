from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="데이터셋 이름")
    description: str = Field(default="", description="설명")
    source: str = Field(default="local", description="출처 (local | roboflow)")


class DatasetUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None


class DatasetRead(BaseModel):
    id: int
    name: str
    description: str
    source: str
    created_at: datetime
    updated_at: datetime
    image_count: int = 0
    annotation_count: int = 0
    class_count: int = 0
    shard_id: Optional[int] = None   # 배정된 샤드 번호

    model_config = {"from_attributes": True}


class DatasetList(BaseModel):
    items: list[DatasetRead]
    total: int
