from sqlalchemy import String, Float, Integer, DateTime, Text, BigInteger, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class OnnxModel(Base):
    __tablename__ = "onnx_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    architecture: Mapped[str] = mapped_column(String(50), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    class_labels: Mapped[str] = mapped_column(Text, nullable=False)   # JSON array
    input_width: Mapped[int] = mapped_column(Integer, nullable=False, default=640)
    input_height: Mapped[int] = mapped_column(Integer, nullable=False, default=640)
    conf_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.25)
    iou_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.45)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
