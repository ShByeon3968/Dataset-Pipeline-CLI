import json
from pathlib import Path

import typer
from rich.table import Table

from cli import client, display

app = typer.Typer(help="ONNX 커스텀 모델 관리")

@app.command("upload")
def upload_onnx_model(
    path: Path = typer.Option(..., "--path", "-p", help="업로드할 .onnx 파일 경로"),
    name: str = typer.Option("", "--name", "-n", help="모델 이름 (비워두면 파일명 사용)"),
    arch: str = typer.Option("yolov8", "--arch", "-a", help="모델 아키텍처 (예: yolov8)"),
    labels: str = typer.Option('[]', "--labels", "-l", help='클래스 라벨 JSON 문자열 (예: \'["person", "car"]\')'),
    width: int = typer.Option(640, "--width", help="입력 이미지 너비"),
    height: int = typer.Option(640, "--height", help="입력 이미지 높이"),
    conf: float = typer.Option(0.25, "--conf", help="기본 신뢰도(Confidence) 임계값"),
    iou: float = typer.Option(0.45, "--iou", help="기본 IoU 임계값"),
):
    """
    새로운 ONNX 커스텀 모델 파일을 업로드하고 시스템에 등록합니다.
    """
    if not path.exists() or not path.is_file():
        display.error(f"파일을 찾을 수 없습니다: {path}")
        raise typer.Exit(1)
        
    if path.suffix.lower() != ".onnx":
        display.error("ONNX 파일(.onnx)만 업로드할 수 있습니다.")
        raise typer.Exit(1)
        
    try:
        # Validate labels JSON
        json.loads(labels)
    except json.JSONDecodeError:
        display.error("클래스 라벨 형식이 올바른 JSON 배열이 아닙니다. 예: '[\"cat\", \"dog\"]'")
        raise typer.Exit(1)
        
    display.info(f"업로드 준비 중: {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    try:
        with open(path, "rb") as f:
            with display.console.status("[bold cyan]모델 업로드 중..."):
                data = {
                    "name": name,
                    "architecture": arch,
                    "class_labels": labels,
                    "input_width": str(width),
                    "input_height": str(height),
                    "conf_threshold": str(conf),
                    "iou_threshold": str(iou),
                }
                result = client.post(
                    "/api/v1/onnx-models/upload",
                    files={"file": (path.name, f, "application/octet-stream")},
                    data=data
                )
    except client.PipelineError as e:
        display.error(str(e))
        raise typer.Exit(1)
        
    display.success(f"ONNX 모델 등록 완료: [bold]{result['name']}[/bold] (ID={result['id']})")
    display.print_kv({
        "ID": result["id"],
        "이름": result["name"],
        "아키텍처": result["architecture"],
        "클래스 목록": ", ".join(result["class_labels"]),
        "입력 크기": f"{result['input_width']}x{result['input_height']}",
    }, title="등록된 모델 정보")


@app.command("list")
def list_onnx_models(
    skip: int = typer.Option(0, "--skip"),
    limit: int = typer.Option(50, "--limit"),
):
    """등록된 ONNX 커스텀 모델 목록을 조회합니다."""
    try:
        result = client.get("/api/v1/onnx-models", params={"skip": skip, "limit": limit})
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    items = result if isinstance(result, list) else result.get("items", [])
    if not items:
        display.info("등록된 ONNX 모델이 없습니다."); return

    rows = [
        [m["id"], m.get("name"), m.get("architecture"), 
         ", ".join(m.get("class_labels", [])) if m.get("class_labels") else "-",
         f"{m.get('input_width')}x{m.get('input_height')}",
         m.get("created_at", "-")[:10] if m.get("created_at") else "-"]
        for m in items
    ]
    display.print_table(
        ["ID", "이름", "아키텍처", "클래스", "입력크기", "등록일"],
        rows, title="ONNX 커스텀 모델 목록"
    )

@app.command("delete")
def delete_onnx_model(
    model_id: int = typer.Option(..., "--id", "-i", help="모델 ID"),
    yes: bool = typer.Option(False, "--yes", "-y", help="확인 없이 삭제"),
):
    """등록된 ONNX 모델을 삭제합니다."""
    if not yes:
        typer.confirm(f"ONNX 모델 {model_id}번을 정말 삭제하시겠습니까?", abort=True)
    try:
        client.delete(f"/api/v1/onnx-models/{model_id}")
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)
    display.success(f"ONNX 모델 {model_id}번 삭제 완료")
