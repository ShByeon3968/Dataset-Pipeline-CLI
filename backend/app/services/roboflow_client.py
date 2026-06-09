"""
Roboflow API 클라이언트
"""
import os
from pathlib import Path

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}


def validate_api_key(api_key: str) -> tuple[bool, str]:
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        _ = rf.workspace()
        return True, "API 키 검증 완료"
    except Exception as e:
        return False, f"API 키 오류: {e}"


def download_dataset(api_key: str, workspace: str, project_id: str, version: int, dest: str) -> str:
    from roboflow import Roboflow
    os.makedirs(dest, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project_id)
    ver = proj.version(version)
    ver.download("coco", location=dest)
    return dest


def get_images_from_roboflow_dir(directory: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if Path(fname).suffix.lower() in SUPPORTED_FORMATS:
                paths.append(os.path.join(root, fname))
    return paths
