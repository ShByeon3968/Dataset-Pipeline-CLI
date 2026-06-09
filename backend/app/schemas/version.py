"""
버저닝 & 리니지 Pydantic 스키마
"""
from __future__ import annotations
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────
# DatasetVersion
# ────────────────────────────────────────────────────────

class DatasetVersionCreate(BaseModel):
    version_name: str = Field(..., max_length=100, description="버전 이름 (예: v1.0.0)")
    description: str = Field("", description="버전 설명")
    created_by: str = Field("user", max_length=200)
    branch_name: str = Field("main", max_length=100)
    parent_version_id: int | None = Field(None, description="부모 버전 ID (브랜치/태그 기반)")
    tags: str = Field("", max_length=200, description="쉼표 구분 태그")


class DatasetVersionRead(BaseModel):
    id: int
    dataset_id: int
    version_name: str
    description: str
    created_by: str
    created_at: datetime
    parent_version_id: int | None
    branch_name: str
    image_count: int
    annotation_count: int
    class_count: int
    added_images: int
    deleted_images: int
    modified_labels: int
    class_distribution: list[dict[str, Any]]
    image_ids_hash: str
    tags: str

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_model(cls, obj: Any) -> "DatasetVersionRead":
        import json
        data = {c.key: getattr(obj, c.key) for c in obj.__table__.columns}
        data["class_distribution"] = json.loads(obj.class_distribution or "[]")
        return cls(**data)


class DatasetVersionList(BaseModel):
    items: list[DatasetVersionRead]
    total: int


# ────────────────────────────────────────────────────────
# ModelVersion
# ────────────────────────────────────────────────────────

class ModelVersionCreate(BaseModel):
    name: str = Field(..., max_length=200)
    description: str = Field("")
    framework: str = Field("", max_length=50, description="예: YOLOv8, Detectron2")
    trained_at: datetime | None = None
    created_by: str = Field("user", max_length=200)


class ModelVersionRead(BaseModel):
    id: int
    name: str
    description: str
    framework: str
    trained_at: datetime | None
    created_at: datetime
    created_by: str

    model_config = {"from_attributes": True}


class ModelVersionList(BaseModel):
    items: list[ModelVersionRead]
    total: int


# ────────────────────────────────────────────────────────
# ModelDatasetLink
# ────────────────────────────────────────────────────────

class ModelDatasetLinkCreate(BaseModel):
    dataset_version_id: int
    dataset_id: int
    linked_by: str = Field("user", max_length=200)
    note: str = Field("")


class ModelDatasetLinkRead(BaseModel):
    id: int
    model_version_id: int
    dataset_version_id: int
    dataset_id: int
    linked_at: datetime
    linked_by: str
    note: str
    is_active: bool

    model_config = {"from_attributes": True}


# ────────────────────────────────────────────────────────
# Lineage graph response
# ────────────────────────────────────────────────────────

class LineageNode(BaseModel):
    id: int
    type: str          # "dataset_version" | "model_version"
    label: str
    dataset_id: int | None = None
    version_name: str | None = None
    branch_name: str | None = None
    framework: str | None = None
    created_at: datetime


class LineageEdge(BaseModel):
    source: int        # source node id
    source_type: str
    target: int
    target_type: str
    label: str = ""


class LineageGraph(BaseModel):
    nodes: list[LineageNode]
    edges: list[LineageEdge]
