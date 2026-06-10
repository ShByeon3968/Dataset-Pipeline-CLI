"""
pipeline autolabel <sub-command>

  run       AI 자동 라벨링 실행 (YOLO-World 또는 ONNX 커스텀 모델)
  status    실행 중인 / 완료된 작업 상태 조회
  list      데이터셋별 실행 이력 목록
  models    등록된 ONNX 커스텀 모델 목록
"""
from __future__ import annotations

import time

import typer
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich import box

from cli import client, display
from cli.config import get_config

app = typer.Typer(help="AI 자동 라벨링")

_POLL_INTERVAL = 2.0   # seconds


def _status_table(run: dict) -> Table:
    """실시간 상태 갱신용 테이블 생성"""
    t = Table(box=box.ROUNDED, show_header=False, expand=False)
    t.add_column("항목",  style="bold cyan", min_width=18)
    t.add_column("값",    style="white")

    processed = run.get("processed_images", 0)
    total     = run.get("total_images", 0) or 1
    pct       = int(processed / total * 100)
    bar       = "█" * (pct // 5) + "░" * (20 - pct // 5)

    t.add_row("Run ID",      str(run.get("id", "-")))
    t.add_row("모델",         run.get("model_name", "-"))
    t.add_row("상태",         run.get("status", "-"))
    t.add_row("진행률",       f"{bar}  {pct}%  ({processed}/{total})")
    t.add_row("생성 어노테이션", str(run.get("total_annotations", 0)))
    if run.get("error_message"):
        t.add_row("오류",     f"[red]{run['error_message']}[/red]")
    return t


@app.command("run")
def run_autolabel(
    dataset_id:  int   = typer.Option(...,     "--dataset-id",  "-d", help="데이터셋 ID"),
    model:       str   = typer.Option("yolo-world", "--model",  "-m",
                                      help="모델 종류: yolo-world | onnx"),
    prompts:     str   = typer.Option("",      "--prompts",     "-p",
                                      help="쉼표 구분 텍스트 프롬프트 (yolo-world 전용)\n예: 'person,car,truck'"),
    onnx_id:     int | None = typer.Option(None, "--onnx-model-id",
                                      help="ONNX 모델 ID (--model onnx 일 때 필수)"),
    conf:        float = typer.Option(None,    "--conf",         help="신뢰도 임계값 (기본: .pipeline.toml)"),
    iou:         float = typer.Option(None,    "--iou",          help="IOU 임계값    (기본: .pipeline.toml)"),
    no_watch:    bool  = typer.Option(False,   "--no-watch",
                                      help="실행 후 완료 대기 없이 즉시 반환"),
):
    """
    AI 자동 라벨링을 실행합니다.

    \b
    # YOLO-World (텍스트 프롬프트 기반 오픈 어휘 탐지)
    pipeline autolabel run --dataset-id 1 --model yolo-world --prompts "person,car,truck"

    \b
    # ONNX 커스텀 모델
    pipeline autolabel run --dataset-id 1 --model onnx --onnx-model-id 2
    """
    cfg = get_config()
    conf = conf if conf is not None else cfg.confidence
    iou  = iou  if iou  is not None else cfg.iou

    # 모델 이름 정규화
    model_name = model.lower().replace("-", "_")
    if model_name == "yolo_world":
        model_name = "yolo-world"

    if model_name == "onnx" and onnx_id is None:
        display.error("--model onnx 사용 시 --onnx-model-id 가 필요합니다.")
        raise typer.Exit(1)

    prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]

    if model_name == "yolo-world" and not prompt_list:
        display.error("yolo-world 모델은 --prompts 가 필요합니다. 예: --prompts 'person,car'")
        raise typer.Exit(1)

    payload: dict = {
        "model_name":           model_name,
        "confidence_threshold": conf,
        "iou_threshold":        iou,
    }
    if prompt_list:
        payload["text_prompts"] = ",".join(prompt_list)
    if onnx_id is not None:
        payload["onnx_model_id"] = onnx_id

    display.info(f"데이터셋 #{dataset_id}  모델: {model_name}  conf={conf}  iou={iou}")
    if prompt_list:
        display.info(f"프롬프트: {', '.join(prompt_list)}")

    try:
        run = client.post(f"/api/v1/datasets/{dataset_id}/auto-label", json=payload)
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    run_id = run["id"]
    display.success(f"자동 라벨링 시작  (run_id={run_id})")

    if no_watch:
        display.info("--no-watch 옵션: 완료 대기를 건너뜁니다.")
        display.info(f"상태 확인: pipeline autolabel status --run-id {run_id}")
        return

    # 진행률 폴링
    with Live(refresh_per_second=2) as live:
        while True:
            try:
                run = client.get(f"/api/v1/auto-label-runs/{run_id}")
            except client.PipelineError as e:
                display.error(str(e)); raise typer.Exit(1)

            live.update(_status_table(run))

            status = run.get("status", "")
            if status == "completed":
                display.success(
                    f"완료  │  {run['total_images']}개 이미지  "
                    f"│  {run['total_annotations']}개 어노테이션 생성"
                )
                break
            elif status == "failed":
                display.error(f"실패: {run.get('error_message', '알 수 없는 오류')}")
                raise typer.Exit(1)

            time.sleep(_POLL_INTERVAL)


@app.command("status")
def run_status(
    run_id: int = typer.Option(..., "--run-id", "-r", help="Run ID"),
):
    """자동 라벨링 작업 상태를 조회합니다."""
    try:
        run = client.get(f"/api/v1/auto-label-runs/{run_id}")
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    processed = run.get("processed_images", 0)
    total     = run.get("total_images", 0) or 1

    display.print_kv({
        "Run ID":       run.get("id"),
        "데이터셋 ID":  run.get("dataset_id"),
        "모델":         run.get("model_name"),
        "상태":         run.get("status"),
        "진행":         f"{processed} / {total}  ({processed / total * 100:.1f}%)",
        "어노테이션":   run.get("total_annotations"),
        "오류":         run.get("error_message") or "-",
        "시작":         run.get("created_at", "-")[:19] if run.get("created_at") else "-",
        "갱신":         run.get("updated_at", "-")[:19] if run.get("updated_at") else "-",
    }, title=f"Auto-label Run #{run_id}")


@app.command("list")
def list_runs(
    dataset_id: int = typer.Option(..., "--dataset-id", "-d", help="데이터셋 ID"),
    limit:      int = typer.Option(20,  "--limit",      "-l", help="최대 반환 수"),
):
    """데이터셋의 자동 라벨링 실행 이력을 조회합니다."""
    try:
        result = client.get(f"/api/v1/datasets/{dataset_id}/auto-label-runs",
                            params={"limit": limit})
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    items = result if isinstance(result, list) else result.get("items", [])
    if not items:
        display.info("실행 이력이 없습니다."); return

    rows = [
        [r["id"], r.get("model_name"), r.get("status"),
         f"{r.get('processed_images', 0)}/{r.get('total_images', 0)}",
         r.get("total_annotations", 0),
         r.get("created_at", "-")[:19] if r.get("created_at") else "-"]
        for r in items
    ]
    display.print_table(
        ["Run ID", "모델", "상태", "진행", "어노테이션", "시작"],
        rows, title=f"데이터셋 #{dataset_id} 자동 라벨링 이력"
    )


@app.command("models")
def list_onnx_models(
    skip:  int = typer.Option(0,  "--skip"),
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
        [m["id"], m.get("name"), m.get("architecture"), m.get("class_labels", "-"),
         m.get("input_width"), m.get("input_height"),
         m.get("created_at", "-")[:10] if m.get("created_at") else "-"]
        for m in items
    ]
    display.print_table(
        ["ID", "이름", "아키텍처", "클래스", "입력W", "입력H", "등록일"],
        rows, title="ONNX 커스텀 모델 목록"
    )
