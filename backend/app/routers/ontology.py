"""
온톨로지 매핑 라우터

- router      : /datasets/{dataset_id}/ontology  → 데이터셋별 샤드 세션 사용
- rules_router: /ontology/rules                  → 전역 규칙이므로 메타 DB(get_db) 사용
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.sharding.router import get_sharded_db, shard_router
from app.models import Class, OntologyRule, OntologyHistory, Dataset
from app.schemas.ontology import (
    OntologyMappingRequest, OntologyRuleCreate, OntologyRuleRead, OntologyHistoryRead
)
from app.services.ontology import apply_class_mapping, undo_ontology_mapping
import json

router = APIRouter(prefix="/datasets/{dataset_id}/ontology", tags=["ontology"])
rules_router = APIRouter(prefix="/ontology/rules", tags=["ontology"])


@router.post("/map")
async def map_classes(
    dataset_id: int,
    req: OntologyMappingRequest,
    db: AsyncSession = Depends(get_sharded_db),
):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다.")

    # 규칙 저장이 필요한 경우, 매핑 전에 소스 클래스 이름을 미리 조회
    # (apply_class_mapping이 소스 클래스를 삭제하기 때문)
    source_names: list[str] = []
    if req.save_as_rule and req.rule_name:
        cls_result = await db.execute(
            select(Class).where(Class.id.in_(req.source_class_ids))
        )
        source_names = [c.name for c in cls_result.scalars().all()]

    # 클래스 매핑 실행
    result = await apply_class_mapping(
        db, dataset_id, req.source_class_ids, req.target_class_name
    )

    # 규칙 저장 — OntologyRule은 전역(메타 DB, shard_0)에 보관
    if req.save_as_rule and req.rule_name and source_names:
        meta_session = shard_router.get_meta_session()
        try:
            rule = OntologyRule(
                name=req.rule_name,
                description=(
                    f"Auto-saved: {', '.join(source_names)} → {req.target_class_name}"
                ),
                rule_data=json.dumps(
                    {"sources": source_names, "target": req.target_class_name},
                    ensure_ascii=False,
                ),
            )
            meta_session.add(rule)
            await meta_session.commit()
            result["saved_rule_name"] = req.rule_name
        except Exception as e:
            await meta_session.rollback()
            # 규칙 저장 실패는 경고로 처리 — 매핑 자체는 이미 성공
            result["rule_save_warning"] = f"규칙 저장 실패: {e}"
        finally:
            await meta_session.close()

    return result


@router.get("/history", response_model=list[OntologyHistoryRead])
async def get_history(dataset_id: int, db: AsyncSession = Depends(get_sharded_db)):
    result = await db.execute(
        select(OntologyHistory)
        .where(OntologyHistory.dataset_id == dataset_id)
        .order_by(OntologyHistory.created_at.desc())
    )
    rows = result.scalars().all()
    return [
        OntologyHistoryRead(
            id=r.id, dataset_id=r.dataset_id, action=r.action,
            before_state=json.loads(r.before_state) if r.before_state else None,
            after_state=json.loads(r.after_state) if r.after_state else None,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.post("/history/{history_id}/undo")
async def undo_history(
    dataset_id: int,
    history_id: int,
    db: AsyncSession = Depends(get_sharded_db),
):
    try:
        return await undo_ontology_mapping(db, history_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"되돌리기 실패: {e}")


# ── 전역 규칙 CRUD ────────────────────────────────────────────────────

@rules_router.get("", response_model=list[OntologyRuleRead])
async def list_rules(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(OntologyRule).order_by(OntologyRule.id))
    return [OntologyRuleRead.model_validate(r) for r in result.scalars()]


@rules_router.post("", response_model=OntologyRuleRead, status_code=status.HTTP_201_CREATED)
async def create_rule(payload: OntologyRuleCreate, db: AsyncSession = Depends(get_db)):
    rule = OntologyRule(
        name=payload.name,
        description=payload.description,
        rule_data=json.dumps(payload.rule_data, ensure_ascii=False)
        if isinstance(payload.rule_data, dict)
        else payload.rule_data,
    )
    db.add(rule)
    await db.commit()
    await db.refresh(rule)
    return OntologyRuleRead.model_validate(rule)


@rules_router.delete("/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rule(rule_id: int, db: AsyncSession = Depends(get_db)):
    rule = await db.get(OntologyRule, rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="규칙을 찾을 수 없습니다.")
    await db.delete(rule)
    await db.commit()
