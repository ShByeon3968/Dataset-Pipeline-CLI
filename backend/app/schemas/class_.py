from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class ClassCreate(BaseModel):
    dataset_id: int
    name: str
    color: Optional[str] = None


class ClassUpdate(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None


class ClassRead(BaseModel):
    id: int
    dataset_id: int
    name: str
    color: str
    created_at: datetime

    model_config = {"from_attributes": True}
