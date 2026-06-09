"""
이미지 임베딩 서비스

추출 모델 (우선순위 순)
  1. CLIP ViT-B/32  — ultralytics fork (git+https://github.com/ultralytics/CLIP.git)
                     512-dim, L2 정규화
  2. 컬러 히스토그램 — 항상 사용 가능한 폴백 (192-dim, R/G/B 각 64 bin)

캐시 정책
  {embeddings_dir}/{dataset_id}/{image_id}.npy 에 numpy float32 배열 저장.
  캐시가 존재하면 모델 추론을 건너뜁니다.

사용 방법
  from app.services.embedding_service import get_or_compute, batch_compute

  # 단일 이미지 (캐시 우선)
  vec = get_or_compute(image_id=3, filepath="2/abc.jpg", dataset_id=2)

  # 데이터셋 전체 (백그라운드 태스크 등에서 호출)
  results = await batch_compute(db, dataset_id=2)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import get_settings
from app.services.file_handler import resolve_filepath

logger = logging.getLogger(__name__)
settings = get_settings()

# ── 프로세스 전역 모델 캐시 ──────────────────────────────────────────
_clip_model = None
_clip_preprocess = None
_clip_device: str = "cpu"


# ── CLIP 로더 ────────────────────────────────────────────────────────

def _load_clip():
    """CLIP 모델을 최초 1회만 로드. 실패하면 None 반환."""
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is not None:
        return _clip_model, _clip_preprocess

    try:
        import clip  # ultralytics/CLIP fork
        import torch

        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_clip_device)
        _clip_model.eval()
        logger.info("CLIP ViT-B/32 loaded on %s", _clip_device)
    except Exception as exc:
        logger.warning("CLIP 로드 실패 — 히스토그램 폴백 사용: %s", exc)
        _clip_model = None
        _clip_preprocess = None

    return _clip_model, _clip_preprocess


# ── 임베딩 추출 ──────────────────────────────────────────────────────

def _embed_clip(abs_path: str) -> Optional[np.ndarray]:
    """CLIP으로 512-dim 벡터 반환. 실패 시 None."""
    model, preprocess = _load_clip()
    if model is None or preprocess is None:
        return None
    try:
        import clip
        import torch
        from PIL import Image as PILImage

        img = PILImage.open(abs_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(_clip_device)
        with torch.no_grad():
            features = model.encode_image(tensor)
        vec = features.cpu().numpy().astype(np.float32).flatten()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    except Exception as exc:
        logger.warning("CLIP 임베딩 실패 (%s): %s", abs_path, exc)
        return None


def _embed_histogram(abs_path: str) -> np.ndarray:
    """컬러 히스토그램 폴백 — 항상 성공 (192-dim)."""
    from PIL import Image as PILImage

    img = PILImage.open(abs_path).convert("RGB").resize((128, 128))
    arr = np.array(img)
    hist: list[int] = []
    for ch in range(3):
        h, _ = np.histogram(arr[:, :, ch], bins=64, range=(0, 256))
        hist.extend(h.tolist())
    vec = np.array(hist, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ── 캐시 경로 ────────────────────────────────────────────────────────

def _cache_path(dataset_id: int, image_id: int) -> Path:
    return Path(settings.embeddings_dir) / str(dataset_id) / f"{image_id}.npy"


def load_cached(dataset_id: int, image_id: int) -> Optional[np.ndarray]:
    p = _cache_path(dataset_id, image_id)
    if p.exists():
        try:
            return np.load(str(p))
        except Exception:
            p.unlink(missing_ok=True)
    return None


def _save(dataset_id: int, image_id: int, vec: np.ndarray) -> None:
    p = _cache_path(dataset_id, image_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(p), vec)


# ── 공개 API ─────────────────────────────────────────────────────────

def get_or_compute(
    image_id: int,
    filepath: str,
    dataset_id: int,
) -> Optional[np.ndarray]:
    """
    캐시된 임베딩 반환. 없으면 CLIP → 히스토그램 순으로 계산 후 캐시.
    파일이 존재하지 않으면 None 반환.
    """
    cached = load_cached(dataset_id, image_id)
    if cached is not None:
        return cached

    abs_path = resolve_filepath(filepath)
    if not Path(abs_path).exists():
        logger.warning("이미지 파일 없음: %s", abs_path)
        return None

    vec = _embed_clip(abs_path)
    if vec is None:
        try:
            vec = _embed_histogram(abs_path)
        except Exception as exc:
            logger.error("히스토그램 폴백도 실패 (%s): %s", abs_path, exc)
            return None

    _save(dataset_id, image_id, vec)
    return vec


async def batch_compute(db, dataset_id: int) -> dict:
    """
    데이터셋의 모든 이미지 임베딩을 계산(캐시 없는 것만).
    백그라운드 태스크에서 호출하세요.

    Returns:
        {"computed": int, "cached": int, "failed": int}
    """
    from sqlalchemy import select
    from app.models.image import Image

    rows = (
        await db.execute(
            select(Image.id, Image.filepath)
            .where(Image.dataset_id == dataset_id)
            .order_by(Image.id)
        )
    ).all()

    computed = cached = failed = 0
    for image_id, filepath in rows:
        if load_cached(dataset_id, image_id) is not None:
            cached += 1
            continue
        vec = get_or_compute(image_id, filepath, dataset_id)
        if vec is not None:
            computed += 1
        else:
            failed += 1

    logger.info(
        "batch_compute dataset=%d  computed=%d cached=%d failed=%d",
        dataset_id, computed, cached, failed,
    )
    return {"computed": computed, "cached": cached, "failed": failed}
