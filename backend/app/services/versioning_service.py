"""
버저닝 서비스
- 버전 생성: 샤드 DB에서 통계 조회 → 메타 DB에 DatasetVersion 저장
- 버전 조회/리스트/삭제
- 리니지 그래프 구성
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime
from sqlalchemy import select, func as F, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.version import DatasetVersion, ModelVersion, ModelDatasetLink
from app.models.image import Image
from app.models.annotation import Annotation
from app.models.class_ import Class
from app.schemas.version import (
    DatasetVersionCreate, DatasetVersionRead,
    ModelVersionCreate, ModelVersionRead,
    ModelDatasetLinkCreate, ModelDatasetLinkRead,
    LineageGraph, LineageNode, LineageEdge,
)


# ─────────────────────────────────────────────────────────────
# 내부 헬퍼: 샤드 DB에서 스냅샷 통계 수집
# ─────────────────────────────────────────────────────────────

async def _collect_snapshot_stats(
    shard_db: AsyncSession, dataset_id: int
) -> dict:
    """샤드 DB에서 현재 데이터셋 상태를 집계."""

    # 이미지 수
    img_count_row = await shard_db.execute(
        select(F.count(Image.id)).where(Image.dataset_id == dataset_id)
    )
    image_count = img_count_row.scalar_one() or 0

    # 어노테이션 수
    ann_count_row = await shard_db.execute(
        select(F.count(Annotation.id))
        .join(Image, Annotation.image_id == Image.id)
        .where(Image.dataset_id == dataset_id)
    )
    annotation_count = ann_count_row.scalar_one() or 0

    # 클래스 수 + 분포
    class_dist_rows = await shard_db.execute(
        select(Class.name, F.count(Annotation.id).label("cnt"))
        .join(Annotation, Annotation.class_id == Class.id, isouter=True)
        .join(Image, Annotation.image_id == Image.id, isouter=True)
        .where(Class.dataset_id == dataset_id)
        .group_by(Class.id, Class.name)
        .order_by(F.count(Annotation.id).desc())
    )
    class_rows = class_dist_rows.all()
    class_count = len(class_rows)
    class_distribution = [{"name": r.name, "count": r.cnt or 0} for r in class_rows]

    # 이미지 ID 해시 (무결성 체크용)
    img_ids_row = await shard_db.execute(
        select(Image.id)
        .where(Image.dataset_id == dataset_id)
        .order_by(Image.id)
    )
    img_ids = [str(r[0]) for r in img_ids_row.all()]
    image_ids_hash = hashlib.md5(",".join(img_ids).encode()).hexdigest()

    return {
        "image_count": image_count,
        "annotation_count": annotation_count,
        "class_count": class_count,
        "class_distribution": class_distribution,
        "image_ids_hash": image_ids_hash,
    }


async def _calc_diff(
    meta_db: AsyncSession,
    dataset_id: int,
    parent_version_id: int | None,
    current_stats: dict,
) -> dict:
    """부모 버전과의 차이 계산."""
    if parent_version_id is None:
        return {"added_images": current_stats["image_count"],
                "deleted_images": 0, "modified_labels": 0}

    parent = await meta_db.get(DatasetVersion, parent_version_id)
    if parent is None:
        return {"added_images": 0, "deleted_images": 0, "modified_labels": 0}

    added = max(0, current_stats["image_count"] - parent.image_count)
    deleted = max(0, parent.image_count - current_stats["image_count"])
    modified = abs(current_stats["annotation_count"] - parent.annotation_count)

    return {"added_images": added, "deleted_images": deleted, "modified_labels": modified}


# ─────────────────────────────────────────────────────────────
# DatasetVersion CRUD
# ─────────────────────────────────────────────────────────────

async def create_dataset_version(
    meta_db: AsyncSession,
    shard_db: AsyncSession,
    dataset_id: int,
    payload: DatasetVersionCreate,
) -> DatasetVersionRead:
    stats = await _collect_snapshot_stats(shard_db, dataset_id)
    diff = await _calc_diff(meta_db, dataset_id, payload.parent_version_id, stats)

    ver = DatasetVersion(
        dataset_id=dataset_id,
        version_name=payload.version_name,
        description=payload.description,
        created_by=payload.created_by,
        branch_name=payload.branch_name,
        parent_version_id=payload.parent_version_id,
        tags=payload.tags,
        image_count=stats["image_count"],
        annotation_count=stats["annotation_count"],
        class_count=stats["class_count"],
        class_distribution=json.dumps(stats["class_distribution"], ensure_ascii=False),
        image_ids_hash=stats["image_ids_hash"],
        **diff,
    )
    meta_db.add(ver)
    await meta_db.flush()
    await meta_db.refresh(ver)
    return DatasetVersionRead.from_orm_model(ver)


async def list_dataset_versions(
    meta_db: AsyncSession,
    dataset_id: int,
    branch: str | None = None,
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[DatasetVersionRead], int]:
    q = select(DatasetVersion).where(DatasetVersion.dataset_id == dataset_id)
    if branch:
        q = q.where(DatasetVersion.branch_name == branch)

    total_row = await meta_db.execute(
        select(F.count()).select_from(q.subquery())
    )
    total = total_row.scalar_one() or 0

    q = q.order_by(DatasetVersion.created_at.desc()).offset(skip).limit(limit)
    rows = (await meta_db.execute(q)).scalars().all()
    return [DatasetVersionRead.from_orm_model(r) for r in rows], total


async def get_dataset_version(
    meta_db: AsyncSession, version_id: int
) -> DatasetVersionRead | None:
    ver = await meta_db.get(DatasetVersion, version_id)
    if ver is None:
        return None
    return DatasetVersionRead.from_orm_model(ver)


async def delete_dataset_version(meta_db: AsyncSession, version_id: int) -> bool:
    ver = await meta_db.get(DatasetVersion, version_id)
    if ver is None:
        return False
    await meta_db.delete(ver)
    return True


# ─────────────────────────────────────────────────────────────
# ModelVersion CRUD
# ─────────────────────────────────────────────────────────────

async def create_model_version(
    meta_db: AsyncSession, payload: ModelVersionCreate
) -> ModelVersionRead:
    mv = ModelVersion(**payload.model_dump())
    meta_db.add(mv)
    await meta_db.flush()
    await meta_db.refresh(mv)
    return ModelVersionRead.model_validate(mv)


async def list_model_versions(
    meta_db: AsyncSession, skip: int = 0, limit: int = 50
) -> tuple[list[ModelVersionRead], int]:
    total = (await meta_db.execute(select(F.count(ModelVersion.id)))).scalar_one() or 0
    rows = (await meta_db.execute(
        select(ModelVersion).order_by(ModelVersion.created_at.desc()).offset(skip).limit(limit)
    )).scalars().all()
    return [ModelVersionRead.model_validate(r) for r in rows], total


async def get_model_version(
    meta_db: AsyncSession, model_version_id: int
) -> ModelVersionRead | None:
    mv = await meta_db.get(ModelVersion, model_version_id)
    return ModelVersionRead.model_validate(mv) if mv else None


async def delete_model_version(meta_db: AsyncSession, model_version_id: int) -> bool:
    mv = await meta_db.get(ModelVersion, model_version_id)
    if mv is None:
        return False
    await meta_db.delete(mv)
    return True


# ─────────────────────────────────────────────────────────────
# ModelDatasetLink CRUD
# ─────────────────────────────────────────────────────────────

async def link_model_to_dataset_version(
    meta_db: AsyncSession,
    model_version_id: int,
    payload: ModelDatasetLinkCreate,
) -> ModelDatasetLinkRead:
    link = ModelDatasetLink(
        model_version_id=model_version_id,
        dataset_version_id=payload.dataset_version_id,
        dataset_id=payload.dataset_id,
        linked_by=payload.linked_by,
        note=payload.note,
        is_active=True,
    )
    meta_db.add(link)
    await meta_db.flush()
    await meta_db.refresh(link)
    return ModelDatasetLinkRead.model_validate(link)


async def list_model_links(
    meta_db: AsyncSession, model_version_id: int
) -> list[ModelDatasetLinkRead]:
    rows = (await meta_db.execute(
        select(ModelDatasetLink)
        .where(ModelDatasetLink.model_version_id == model_version_id,
               ModelDatasetLink.is_active == True)
        .order_by(ModelDatasetLink.linked_at.desc())
    )).scalars().all()
    return [ModelDatasetLinkRead.model_validate(r) for r in rows]


async def unlink_model_from_dataset_version(
    meta_db: AsyncSession, link_id: int
) -> bool:
    link = await meta_db.get(ModelDatasetLink, link_id)
    if link is None:
        return False
    link.is_active = False
    return True


# ─────────────────────────────────────────────────────────────
# Lineage graph (dataset 기준)
# ─────────────────────────────────────────────────────────────

async def build_lineage_graph(
    meta_db: AsyncSession, dataset_id: int
) -> LineageGraph:
    """특정 dataset_id에 연결된 모든 버전 + 모델 노드/엣지 반환."""

    # 데이터셋 버전 노드
    dv_rows = (await meta_db.execute(
        select(DatasetVersion)
        .where(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.created_at)
    )).scalars().all()

    nodes: list[LineageNode] = []
    edges: list[LineageEdge] = []

    for dv in dv_rows:
        nodes.append(LineageNode(
            id=dv.id,
            type="dataset_version",
            label=f"{dv.version_name} [{dv.branch_name}]",
            dataset_id=dv.dataset_id,
            version_name=dv.version_name,
            branch_name=dv.branch_name,
            created_at=dv.created_at,
        ))
        if dv.parent_version_id is not None:
            edges.append(LineageEdge(
                source=dv.parent_version_id,
                source_type="dataset_version",
                target=dv.id,
                target_type="dataset_version",
                label="parent",
            ))

    # 연결된 모델 버전
    dv_ids = [dv.id for dv in dv_rows]
    if dv_ids:
        link_rows = (await meta_db.execute(
            select(ModelDatasetLink)
            .where(
                ModelDatasetLink.dataset_version_id.in_(dv_ids),
                ModelDatasetLink.is_active == True,
            )
        )).scalars().all()

        mv_ids_seen: set[int] = set()
        for link in link_rows:
            if link.model_version_id not in mv_ids_seen:
                mv = await meta_db.get(ModelVersion, link.model_version_id)
                if mv:
                    nodes.append(LineageNode(
                        id=mv.id,
                        type="model_version",
                        label=f"{mv.name} ({mv.framework})" if mv.framework else mv.name,
                        framework=mv.framework,
                        created_at=mv.created_at,
                    ))
                    mv_ids_seen.add(mv.id)

            edges.append(LineageEdge(
                source=link.dataset_version_id,
                source_type="dataset_version",
                target=link.model_version_id,
                target_type="model_version",
                label=link.note or "학습에 사용",
            ))

    return LineageGraph(nodes=nodes, edges=edges)
