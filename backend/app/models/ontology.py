from datetime import datetime
from sqlalchemy import String, Text, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class OntologyRule(Base):
    __tablename__ = "ontology_rules"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    rule_data: Mapped[str | None] = mapped_column(Text)  # JSON: {"sources": [...], "target": "..."}
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class OntologyHistory(Base):
    __tablename__ = "ontology_history"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    action: Mapped[str | None] = mapped_column(String(500))
    before_state: Mapped[str | None] = mapped_column(Text)   # JSON 스냅샷
    after_state: Mapped[str | None] = mapped_column(Text)    # JSON 스냅샷
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    dataset = relationship("Dataset", back_populates="ontology_histories")
