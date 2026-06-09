from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class OntologyMappingRequest(BaseModel):
    dataset_id: int
    source_class_ids: list[int]
    target_class_name: str
    save_as_rule: bool = False
    rule_name: Optional[str] = None
    rule_description: Optional[str] = None


class OntologyRuleCreate(BaseModel):
    name: str
    description: str = ""
    sources: list[str]
    target: str


class OntologyRuleRead(BaseModel):
    id: int
    name: str
    description: str
    rule_data: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class OntologyHistoryRead(BaseModel):
    id: int
    dataset_id: int
    action: Optional[str]
    before_state: Optional[dict]
    after_state: Optional[dict]
    created_at: datetime

    model_config = {"from_attributes": True}
