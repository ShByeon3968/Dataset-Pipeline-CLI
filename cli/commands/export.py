"""
pipeline export <sub-command>

  coco    COCO JSON 포맷으로 내보내기
  yolo    YOLO (txt + data.yaml) 포맷으로 내보내기
  voc     Pascal VOC XML 포맷으로 내보내기
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import typer

from cli import client, display
from cli.config import get_config

app = typer.Typer(help="데이터셋 내보내기")


def _do_export(
    dataset_id: int,
    fmt: str,
    out: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    """공통 내보내기 로직: API 호출 → ZIP 다운로드 → 압축 해제"""
    out.mkdir(parents=True, exist_ok=True)
    zip_dest = out / f"export_{dataset_id}_{fmt}.zip"

    endpoint_map = {
        "coco": f"/api/v1/datasets/{dataset_id}/export/coco",
        "yolo": f"/api/v1/datasets/{dataset_id}/export/yolo",
        "voc":  f"/api/v1/datasets/{dataset_id}/export/pascal-voc",
    }
    params = {
        "train_ratio": train_ratio,
        "val_ratio":   val_ratio,
        "test_ratio":  test_ratio,
    }

    with display.console.status(f"[bold cyan]{fmt.upper()} 내보내기 중…"):
        try:
            client.download(endpoint_map[fmt], dest=zip_dest, params=params)
        except client.PipelineError as e:
            display.error(str(e)); raise typer.Exit(1)

    # ZIP 압축 해제
    extract_dir = out / f"dataset_{dataset_id}_{fmt}"
    with zipfile.ZipFile(zip_dest, "r") as zf:
        zf.extractall(extract_dir)

    zip_dest.unlink()  # 임시 ZIP 삭제

    display.success(f"내보내기 완료: [bold]{extract_dir}[/bold]")
    display.print_kv({
        "포맷":        fmt.upper(),
        "데이터셋 ID": dataset_id,
        "저장 경로":   str(extract_dir.resolve()),
        "train 비율":  train_ratio,
        "val 비율":    val_ratio,
        "test 비율":   test_ratio,
    }, title="내보내기 결과")


@app.command("coco")
def export_coco(
    dataset_id:  int   = typer.Option(...,  "--dataset-id",  "-d", help="데이터셋 ID"),
    out:         Path  = typer.Option(Path("."), "--out",    "-o", help="저장 디렉토리"),
    train_ratio: float = typer.Option(None, "--train",             help="train 비율"),
    val_ratio:   float = typer.Option(None, "--val",               help="val 비율"),
    test_ratio:  float = typer.Option(None, "--test",              help="test 비율"),
):
    """COCO JSON 포맷으로 내보냅니다."""
    cfg = get_config()
    _do_export(dataset_id, "coco", out,
               train_ratio or cfg.train_ratio,
               val_ratio   or cfg.val_ratio,
               test_ratio  or cfg.test_ratio)


@app.command("yolo")
def export_yolo(
    dataset_id:  int   = typer.Option(...,  "--dataset-id",  "-d", help="데이터셋 ID"),
    out:         Path  = typer.Option(Path("."), "--out",    "-o", help="저장 디렉토리"),
    train_ratio: float = typer.Option(None, "--train",             help="train 비율"),
    val_ratio:   float = typer.Option(None, "--val",               help="val 비율"),
    test_ratio:  float = typer.Option(None, "--test",              help="test 비율"),
):
    """YOLO 포맷 (txt labels + data.yaml) 으로 내보냅니다."""
    cfg = get_config()
    _do_export(dataset_id, "yolo", out,
               train_ratio or cfg.train_ratio,
               val_ratio   or cfg.val_ratio,
               test_ratio  or cfg.test_ratio)


@app.command("voc")
def export_voc(
    dataset_id:  int   = typer.Option(...,  "--dataset-id",  "-d", help="데이터셋 ID"),
    out:         Path  = typer.Option(Path("."), "--out",    "-o", help="저장 디렉토리"),
    train_ratio: float = typer.Option(None, "--train",             help="train 비율"),
    val_ratio:   float = typer.Option(None, "--val",               help="val 비율"),
    test_ratio:  float = typer.Option(None, "--test",              help="test 비율"),
):
    """Pascal VOC XML 포맷으로 내보냅니다."""
    cfg = get_config()
    _do_export(dataset_id, "voc", out,
               train_ratio or cfg.train_ratio,
               val_ratio   or cfg.val_ratio,
               test_ratio  or cfg.test_ratio)
