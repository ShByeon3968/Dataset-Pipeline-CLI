"""
CLI 설정 관리

우선순위 (높은 순):
  1. 환경변수  PIPELINE_BASE_URL, PIPELINE_TIMEOUT 등
  2. 프로젝트 루트 .pipeline.toml
  3. 홈 디렉토리  ~/.pipeline.toml
  4. 내장 기본값
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # pip install tomli
    except ImportError:
        tomllib = None  # type: ignore


_DEFAULT_BASE_URL = "http://localhost:8001"
_DEFAULT_TIMEOUT  = 120.0


def _load_toml(path: Path) -> dict:
    if tomllib is None or not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


@lru_cache(maxsize=1)
def get_config() -> "Config":
    # 탐색 순서: 현재 디렉토리 → 홈
    project_cfg = _load_toml(Path(".pipeline.toml"))
    home_cfg    = _load_toml(Path.home() / ".pipeline.toml")

    merged = {**home_cfg, **project_cfg}
    api_section      = merged.get("api", {})
    defaults_section = merged.get("defaults", {})

    return Config(
        base_url=os.getenv("PIPELINE_BASE_URL",
                           api_section.get("base_url", _DEFAULT_BASE_URL)).rstrip("/"),
        timeout=float(os.getenv("PIPELINE_TIMEOUT",
                                api_section.get("timeout", _DEFAULT_TIMEOUT))),
        train_ratio=float(defaults_section.get("train_ratio", 0.7)),
        val_ratio=float(defaults_section.get("val_ratio",   0.2)),
        test_ratio=float(defaults_section.get("test_ratio",  0.1)),
        confidence=float(defaults_section.get("confidence",  0.25)),
        iou=float(defaults_section.get("iou", 0.45)),
    )


class Config:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        confidence: float,
        iou: float,
    ):
        self.base_url    = base_url
        self.timeout     = timeout
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.test_ratio  = test_ratio
        self.confidence  = confidence
        self.iou         = iou

    def api(self, path: str) -> str:
        """전체 API URL 조합: /api/v1/datasets → http://host/api/v1/datasets"""
        return f"{self.base_url}{path}"
