"""
백엔드 API HTTP 클라이언트

httpx 동기 클라이언트를 사용합니다.
모든 응답 오류는 PipelineError로 통일합니다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx

from cli.config import get_config


class PipelineError(Exception):
    """API 호출 실패 시 발생하는 예외"""
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _client() -> httpx.Client:
    cfg = get_config()
    return httpx.Client(base_url=cfg.base_url, timeout=cfg.timeout)


def _raise(resp: httpx.Response) -> None:
    if resp.is_error:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise PipelineError(f"HTTP {resp.status_code}: {detail}", resp.status_code)


# ── 공통 CRUD 헬퍼 ────────────────────────────────────────────────────

def get(path: str, params: dict | None = None) -> Any:
    with _client() as c:
        resp = c.get(path, params=params)
    _raise(resp)
    return resp.json()


def post(path: str, json: dict | None = None, data: dict | None = None,
         files: dict | None = None) -> Any:
    with _client() as c:
        resp = c.post(path, json=json, data=data, files=files)
    _raise(resp)
    return resp.json()


def patch(path: str, json: dict) -> Any:
    with _client() as c:
        resp = c.patch(path, json=json)
    _raise(resp)
    return resp.json()


def delete(path: str) -> Any:
    with _client() as c:
        resp = c.delete(path)
    _raise(resp)
    try:
        return resp.json()
    except Exception:
        return {}


def download(path: str, dest: Path, params: dict | None = None) -> Path:
    """파일 다운로드 (스트리밍). dest 경로에 저장 후 반환."""
    cfg = get_config()
    with httpx.stream("GET", cfg.api(path), params=params,
                      timeout=cfg.timeout) as resp:
        if resp.is_error:
            raise PipelineError(f"HTTP {resp.status_code} 다운로드 실패", resp.status_code)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=8192):
                f.write(chunk)
    return dest


def health() -> dict:
    return get("/api/health")
