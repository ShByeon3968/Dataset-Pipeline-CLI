"""
파일 업로드, 저장, 해시 처리 서비스

경로 정책
---------
DB의 Image.filepath 컬럼에는 uploads_dir 기준 상대경로를 저장합니다.
예) "{dataset_id}/{hash_stem}.jpg"

실제 파일에 접근할 때는 resolve_filepath() 를 통해 절대경로를 얻습니다.

NAS 연동 방법
-------------
docker-compose.yml 의 volumes 에서 NAS 마운트 경로를 /app/data/uploads 에
bind mount 하면 uploads_dir 기본값 그대로 NAS에 저장됩니다.
코드 변경 없이 docker-compose.yml / .env 설정만으로 NAS 전환이 가능합니다.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import shutil
import zipfile
from pathlib import Path

from PIL import Image as PILImage

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}


# 경로 유틸

def resolve_filepath(path: str) -> str:
    """
    DB에 저장된 상대경로를 절대경로로 변환.
    - 절대경로로 시작하면 그대로 반환 (하위 호환)
    - 상대경로면 uploads_dir 과 조합해 절대경로 반환
    """
    if os.path.isabs(path):
        return path
    return os.path.join(settings.uploads_dir, path)


def to_relative_path(abs_path: str) -> str:
    """
    절대경로를 uploads_dir 기준 상대경로로 변환.
    예) /app/data/uploads/3/abc_img.jpg → 3/abc_img.jpg
    """
    uploads_abs = os.path.abspath(settings.uploads_dir)
    abs_path_norm = os.path.abspath(abs_path)
    try:
        return os.path.relpath(abs_path_norm, uploads_abs)
    except ValueError:
        return abs_path


# 디렉터리 헬퍼

def get_dataset_upload_dir(dataset_id: int) -> str:
    """데이터셋별 업로드 디렉터리 경로 반환 및 생성."""
    d = os.path.join(settings.uploads_dir, str(dataset_id))
    os.makedirs(d, exist_ok=True)
    return d


# 파일 읽기

def get_file_bytes(path: str) -> bytes | None:
    """로컬 디스크(또는 마운트된 NAS)에서 파일 bytes 반환. 없으면 None."""
    abs_path = resolve_filepath(path)
    try:
        with open(abs_path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


# 해시

def calculate_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def calculate_md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def calculate_phash(filepath: str) -> str:
    """절대경로를 받아 퍼셉추얼 해시 반환."""
    try:
        import imagehash
        img = PILImage.open(filepath).convert("RGB")
        return str(imagehash.phash(img))
    except Exception:
        return ""


# 이미지 메타

def get_image_dimensions(filepath: str) -> tuple:
    with PILImage.open(filepath) as img:
        return img.width, img.height


def get_file_format(filepath: str) -> str:
    return Path(filepath).suffix.lower().lstrip(".")


# 파일 저장

def save_image_to_storage(data: bytes, filename: str, dataset_id: int) -> tuple[str, str]:
    """
    이미지 bytes 를 저장.

    Returns (abs_path, storage_key)
      abs_path     : 크기/해시 추출용 로컬 절대경로
      storage_key  : DB Image.filepath 에 저장할 상대경로
    """
    md5_prefix = calculate_md5_bytes(data)[:8]
    safe_name = f"{md5_prefix}_{Path(filename).name}"
    key = f"{dataset_id}/{safe_name}"
    abs_path = os.path.join(settings.uploads_dir, key)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "wb") as f:
        f.write(data)
    return abs_path, key


def save_uploaded_file(file_bytes: bytes, filename: str, dataset_id: int) -> str:
    """업로드된 파일을 저장하고 상대경로(storage key) 반환."""
    _, key = save_image_to_storage(file_bytes, filename, dataset_id)
    return key


def extract_zip_and_get_images(zip_bytes: bytes, dataset_id: int) -> list:
    """ZIP 파일에서 이미지를 추출하고 storage key 목록 반환."""
    keys = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            suffix = Path(member.filename).suffix.lower()
            if suffix not in SUPPORTED_FORMATS:
                continue
            data = zf.read(member)
            safe_name = Path(member.filename).name
            _, key = save_image_to_storage(data, safe_name, dataset_id)
            keys.append(key)
    return keys


# 파일 삭제

def delete_file(path: str):
    """로컬 디스크(또는 마운트된 NAS)에서 파일 삭제."""
    try:
        abs_path = resolve_filepath(path)
        if os.path.exists(abs_path):
            os.remove(abs_path)
    except Exception:
        pass


# 초기화

def ensure_dirs():
    """앱 시작 시 필요한 디렉터리 생성."""
    os.makedirs(settings.uploads_dir, exist_ok=True)
    os.makedirs(settings.exports_dir, exist_ok=True)
    os.makedirs(settings.embeddings_dir, exist_ok=True)
    os.makedirs(settings.models_dir, exist_ok=True)
