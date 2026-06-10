"""
pipeline upload <sub-command>

  images    이미지 파일(들) 업로드 (단일 or 디렉토리)
  zip       ZIP 파일 업로드 (YOLO / COCO / split 디렉토리 구조 지원)
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

import typer
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from cli import client, display

app = typer.Typer(help="이미지 / ZIP 업로드")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@app.command("images")
def upload_images(
    dataset_id: int  = typer.Option(..., "--dataset-id", "-d", help="대상 데이터셋 ID"),
    path: Path       = typer.Option(..., "--path",       "-p", help="이미지 파일 또는 디렉토리"),
    recursive: bool  = typer.Option(True, "--recursive/--no-recursive", help="하위 디렉토리 포함"),
):
    """이미지 파일을 업로드합니다. 디렉토리를 지정하면 내부 이미지를 모두 업로드합니다."""
    if not path.exists():
        display.error(f"경로가 존재하지 않습니다: {path}"); raise typer.Exit(1)

    if path.is_file():
        files = [path]
    else:
        pattern = "**/*" if recursive else "*"
        files = [f for f in path.glob(pattern) if f.suffix.lower() in IMAGE_EXTS]

    if not files:
        display.warn("업로드할 이미지 파일이 없습니다."); return

    display.info(f"업로드 대상: {len(files)}개 이미지 → 데이터셋 #{dataset_id}")

    success_count = 0
    fail_count    = 0

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("업로드 중", total=len(files))

        for img_path in files:
            mime = mimetypes.guess_type(str(img_path))[0] or "image/jpeg"
            try:
                with open(img_path, "rb") as f:
                    client.post(
                        f"/api/v1/datasets/{dataset_id}/images/upload",
                        files={"files": (img_path.name, f, mime)},
                    )
                success_count += 1
            except client.PipelineError as e:
                fail_count += 1
                display.warn(f"{img_path.name}: {e}")
            progress.advance(task)

    display.success(f"완료: {success_count}개 성공, {fail_count}개 실패")


@app.command("zip")
def upload_zip(
    dataset_id: int  = typer.Option(..., "--dataset-id", "-d", help="대상 데이터셋 ID"),
    path: Path       = typer.Option(..., "--path",       "-p", help="ZIP 파일 경로"),
    format: str      = typer.Option("auto", "--format",  "-f",
                                    help="포맷 힌트: auto | yolo | coco | plain"),
):
    """
    ZIP 파일을 업로드합니다.

    지원 ZIP 구조:
      - train/images/, val/images/, test/images/ (split 자동 인식)
      - images/ + labels/ (YOLO)
      - images/ + _annotations.coco.json (COCO)
      - 이미지만 포함된 평탄(plain) ZIP
    """
    if not path.exists() or not path.is_file():
        display.error(f"ZIP 파일이 존재하지 않습니다: {path}"); raise typer.Exit(1)

    display.info(f"ZIP 업로드: {path.name}  ({path.stat().st_size / 1024 / 1024:.1f} MB)  → 데이터셋 #{dataset_id}")

    try:
        with open(path, "rb") as f:
            with display.console.status("[bold cyan]업로드 중…"):
                result = client.post(
                    f"/api/v1/datasets/{dataset_id}/images/upload-zip",
                    files={"file": (path.name, f, "application/zip")},
                    data={"format_hint": format},
                )
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    display.success("ZIP 업로드 완료")
    display.print_kv({
        "업로드된 이미지": result.get("uploaded_count", result.get("count", "-")),
        "스킵(중복)": result.get("skipped_count", "-"),
        "어노테이션": result.get("annotation_count", "-"),
    }, title="업로드 결과")
