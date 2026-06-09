"""
데이터셋 버저닝 및 모델 리니지 모델
- DatasetVersion : 데이터셋 상태 스냅샷
- ModelVersion   : ML 모델 버전 등록부
- ModelDatasetLink : 모델↔데이터셋 버전 연결(태그)
"""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import (
    String, Text, Integer, Boolean,
    DateTime, ForeignKey, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    version_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(String(200), default="user")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # 브랜치 구조
    parent_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="SET NULL"), nullable=True
    )
    branch_name: Mapped[str] = mapped_column(String(100), default="main")

    # 스냅샷 통계
    image_count: Mapped[int] = mapped_column(Integer, default=0)
    annotation_count: Mapped[int] = mapped_column(Integer, default=0)
    class_count: Mapped[int] = mapped_column(Integer, default=0)

    # 부모 대비 변경 요약
    added_images: Mapped[int] = mapped_column(Integer, default=0)
    deleted_images: Mapped[int] = mapped_column(Integer, default=0)
    modified_labels: Mapped[int] = mapped_column(Integer, default=0)

    # 클래스 분포 스냅샷 (JSON 문자열)
    class_distribution: Mapped[str] = mapped_column(Text, default="[]")

    # 무결성 체크용: 스냅샷 시점의 이미지 ID 해시 (MD5)
    image_ids_hash: Mapped[str] = mapped_column(String(64), default="")

    # 태그 (예: "학습용", "검증용")
    tags: Mapped[str] = mapped_column(String(200), default="")

    # 관계
    parent = relationship("DatasetVersion", remote_side=[id], foreign_keys=[parent_version_id])
    model_links = relationship(
        "ModelDatasetLink", back_populates="dataset_version", cascade="all, delete-orphan"
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    framework: Mapped[str] = mapped_column(String(50), default="")   # e.g. YOLOv8, Detectron2
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    created_by: Mapped[str] = mapped_column(String(200), default="user")

    dataset_links = relationship(
        "ModelDatasetLink", back_populates="model_version", cascade="all, delete-orphan"
    )


class ModelDatasetLink(Base):
    __tablename__ = "model_dataset_links"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_id: Mapped[int] = mapped_column(Integer, nullable=False)  # 역참조 편의용 역정규화
    linked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    linked_by: Mapped[str] = mapped_column(String(200), default="user")
    note: Mapped[str] = mapped_column(Text, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    model_version = relationship("ModelVersion", back_populates="dataset_links")
    dataset_version = relationship("DatasetVersion", back_populates="model_links")
