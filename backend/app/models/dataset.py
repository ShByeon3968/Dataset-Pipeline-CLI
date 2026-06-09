from datetime import datetime
from sqlalchemy import String, Text, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(50), default="local")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # lazy="noload": do NOT auto-load on query; DB-level cascade (ondelete="CASCADE")
    # handles actual deletion. Manual count queries in routers work fine.
    images = relationship(
        "Image", back_populates="dataset",
        cascade="all, delete-orphan", lazy="noload",
    )
    classes = relationship(
        "Class", back_populates="dataset",
        cascade="all, delete-orphan", lazy="noload",
    )
    ontology_histories = relationship(
        "OntologyHistory", back_populates="dataset",
        cascade="all, delete-orphan", lazy="noload",
    )
