#!/usr/bin/env python3
"""
Dataset Pipeline CLI — 진입점

사용법:
  python pipeline.py --help
  python pipeline.py dataset list
  python pipeline.py upload zip --dataset-id 1 --path ./data.zip
  python pipeline.py autolabel run --dataset-id 1 --prompts "person,car"
  python pipeline.py export yolo --dataset-id 1 --out ./output/
"""
import typer

from cli.commands import (
    dataset   as dataset_cmd,
    upload    as upload_cmd,
    autolabel as autolabel_cmd,
    analysis  as analysis_cmd,
    refinement as refinement_cmd,
    export    as export_cmd,
    version   as version_cmd,
)
from cli import client, display

app = typer.Typer(
    name="pipeline",
    help="Dataset Pipeline CLI — ML 데이터셋 구축·분석·정제·내보내기",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

app.add_typer(dataset_cmd.app,    name="dataset",    help="데이터셋 관리")
app.add_typer(upload_cmd.app,     name="upload",     help="이미지 / ZIP 업로드")
app.add_typer(autolabel_cmd.app,  name="autolabel",  help="AI 자동 라벨링")
app.add_typer(analysis_cmd.app,   name="analysis",   help="데이터셋 분석")
app.add_typer(refinement_cmd.app, name="refinement", help="데이터 정제")
app.add_typer(export_cmd.app,     name="export",     help="데이터셋 내보내기")
app.add_typer(version_cmd.app,    name="version",    help="버저닝 및 모델 리니지")


@app.command("health")
def health():
    """백엔드 서버 상태를 확인합니다."""
    from cli.config import get_config
    cfg = get_config()
    display.info(f"백엔드 주소: {cfg.base_url}")
    try:
        result = client.health()
    except Exception as e:
        display.error(f"연결 실패: {e}")
        raise typer.Exit(1)

    status = result.get("status", "unknown")
    db     = result.get("db", "unknown")
    ver    = result.get("version", "-")

    if status == "ok":
        display.success(f"백엔드 정상  │  DB: {db}  │  version: {ver}")
    else:
        display.warn(f"백엔드 degraded  │  DB: {db}  │  version: {ver}")


@app.command("config")
def show_config():
    """현재 적용된 설정을 출력합니다."""
    from cli.config import get_config
    cfg = get_config()
    display.print_kv({
        "base_url":    cfg.base_url,
        "timeout":     f"{cfg.timeout}s",
        "train_ratio": cfg.train_ratio,
        "val_ratio":   cfg.val_ratio,
        "test_ratio":  cfg.test_ratio,
        "confidence":  cfg.confidence,
        "iou":         cfg.iou,
    }, title="현재 설정")
    display.info("설정 파일: .pipeline.toml (프로젝트) 또는 ~/.pipeline.toml (전역)")
    display.info("환경변수: PIPELINE_BASE_URL, PIPELINE_TIMEOUT")


if __name__ == "__main__":
    app()
