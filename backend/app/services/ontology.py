"""
온톨로지 매핑 서비스
"""
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from app.models import Class, Annotation, OntologyRule, OntologyHistory


async def apply_class_mapping(
    db: AsyncSession,
    dataset_id: int,
    source_class_ids: list[int],
    target_class_name: str,
) -> dict:
    """
    여러 소스 클래스를 하나의 타겟 클래스로 매핑.
    - 타겟 클래스가 없으면 새로 생성
    - Annotation의 class_id를 일괄 업데이트
    - 매핑 전/후 상태를 히스토리에 저장
    """
    # 소스 클래스 조회
    stmt = select(Class).where(
        Class.id.in_(source_class_ids),
        Class.dataset_id == dataset_id,
    )
    result = await db.execute(stmt)
    source_classes = result.scalars().all()
    if not source_classes:
        raise ValueError("소스 클래스를 찾을 수 없습니다.")

    # before 상태 저장
    before_state = {
        "classes": [{"id": c.id, "name": c.name} for c in source_classes],
    }

    # 타겟 클래스 조회 또는 생성
    tgt_stmt = select(Class).where(
        Class.dataset_id == dataset_id, Class.name == target_class_name
    )
    tgt_result = await db.execute(tgt_stmt)
    target_class = tgt_result.scalar_one_or_none()

    if target_class is None:
        from app.services.class_service import CLASS_COLORS
        count_stmt = select(Class).where(Class.dataset_id == dataset_id)
        count_result = await db.execute(count_stmt)
        count = len(count_result.scalars().all())
        color = CLASS_COLORS[count % len(CLASS_COLORS)]
        target_class = Class(dataset_id=dataset_id, name=target_class_name, color=color)
        db.add(target_class)
        await db.flush()

    # Annotation 일괄 업데이트
    affected = 0
    for src_id in source_class_ids:
        if src_id == target_class.id:
            continue
        res = await db.execute(
            update(Annotation)
            .where(Annotation.class_id == src_id)
            .values(class_id=target_class.id)
        )
        affected += res.rowcount

    # 소스 클래스 삭제 (타겟 제외)
    ids_to_delete = [c.id for c in source_classes if c.id != target_class.id]
    if ids_to_delete:
        await db.execute(delete(Class).where(Class.id.in_(ids_to_delete)))

    after_state = {
        "target_class": {"id": target_class.id, "name": target_class.name},
        "affected_annotations": affected,
    }

    # 히스토리 저장
    action_desc = (
        f"{', '.join(c.name for c in source_classes)} → {target_class_name}"
    )
    history = OntologyHistory(
        dataset_id=dataset_id,
        action=action_desc,
        before_state=json.dumps(before_state, ensure_ascii=False),
        after_state=json.dumps(after_state, ensure_ascii=False),
    )
    db.add(history)
    await db.commit()

    return {
        "target_class_id": target_class.id,
        "target_class_name": target_class.name,
        "affected_annotations": affected,
        "deleted_class_ids": ids_to_delete,
    }


async def undo_ontology_mapping(db: AsyncSession, history_id: int) -> dict:
    """온톨로지 히스토리 되돌리기"""
    stmt = select(OntologyHistory).where(OntologyHistory.id == history_id)
    result = await db.execute(stmt)
    history = result.scalar_one_or_none()
    if not history:
        raise ValueError("히스토리를 찾을 수 없습니다.")

    before = json.loads(history.before_state)
    after = json.loads(history.after_state)

    # 타겟 클래스 찾기
    target_id = after.get("target_class", {}).get("id")
    if not target_id:
        raise ValueError("되돌릴 수 없는 히스토리입니다.")

    # 소스 클래스 복원
    restored = []
    for cls_data in before.get("classes", []):
        stmt2 = select(Class).where(Class.id == cls_data["id"])
        r2 = await db.execute(stmt2)
        cls = r2.scalar_one_or_none()
        if not cls:
            # 재생성
            cls = Class(
                id=cls_data["id"],
                dataset_id=history.dataset_id,
                name=cls_data["name"],
                color="#CCCCCC",
            )
            db.add(cls)
            await db.flush()
        restored.append(cls_data["id"])

    await db.delete(history)
    await db.commit()

    return {"restored_class_ids": restored}
