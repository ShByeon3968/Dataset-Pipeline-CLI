"""
pipeline dataset <sub-command>

  list            데이터셋 목록 조회
  create          새 데이터셋 생성
  info            데이터셋 상세 정보
  delete          데이터셋 삭제
"""
import typer
from cli import client, display

app = typer.Typer(help="데이터셋 관리")


@app.command("list")
def list_datasets(
    skip:  int = typer.Option(0,   "--skip",  "-s", help="건너뛸 항목 수"),
    limit: int = typer.Option(50,  "--limit", "-l", help="최대 반환 수"),
):
    """데이터셋 목록을 조회합니다."""
    try:
        result = client.get("/api/v1/datasets", params={"skip": skip, "limit": limit})
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    items = result if isinstance(result, list) else result.get("items", result)
    if not items:
        display.info("등록된 데이터셋이 없습니다.")
        return

    rows = [
        [d["id"], d["name"], d.get("source", "-"),
         d.get("image_count", "-"), d.get("annotation_count", "-"),
         d.get("created_at", "-")[:10] if d.get("created_at") else "-"]
        for d in items
    ]
    display.print_table(
        ["ID", "이름", "소스", "이미지", "어노테이션", "생성일"],
        rows, title="데이터셋 목록"
    )


@app.command("create")
def create_dataset(
    name: str        = typer.Option(..., "--name", "-n",  help="데이터셋 이름"),
    desc: str        = typer.Option("",  "--desc", "-d",  help="설명"),
    source: str      = typer.Option("local", "--source",  help="소스 유형 (local / roboflow / ...)"),
):
    """새 데이터셋을 생성합니다."""
    try:
        result = client.post("/api/v1/datasets", json={
            "name": name, "description": desc, "source": source
        })
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    display.success(f"데이터셋 생성 완료: [bold]{result['name']}[/bold]  (id={result['id']})")
    display.print_kv({
        "ID": result["id"],
        "이름": result["name"],
        "설명": result.get("description", "-"),
        "소스": result.get("source", "-"),
        "생성일": result.get("created_at", "-"),
    }, title="생성된 데이터셋")


@app.command("info")
def dataset_info(
    dataset_id: int = typer.Option(..., "--id", "-i", help="데이터셋 ID"),
):
    """데이터셋 상세 정보를 조회합니다."""
    try:
        d = client.get(f"/api/v1/datasets/{dataset_id}")
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    display.print_kv({
        "ID": d["id"],
        "이름": d["name"],
        "설명": d.get("description", "-"),
        "소스": d.get("source", "-"),
        "이미지 수": d.get("image_count", "-"),
        "어노테이션 수": d.get("annotation_count", "-"),
        "클래스 수": d.get("class_count", "-"),
        "생성일": d.get("created_at", "-"),
        "수정일": d.get("updated_at", "-"),
    }, title=f"데이터셋 #{dataset_id}")


@app.command("delete")
def delete_dataset(
    dataset_id: int = typer.Option(..., "--id", "-i", help="데이터셋 ID"),
    yes: bool       = typer.Option(False, "--yes", "-y", help="확인 없이 삭제"),
):
    """데이터셋을 삭제합니다 (이미지·어노테이션 포함)."""
    if not yes:
        typer.confirm(f"데이터셋 {dataset_id}을(를) 정말 삭제하시겠습니까?", abort=True)
    try:
        client.delete(f"/api/v1/datasets/{dataset_id}")
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)
    display.success(f"데이터셋 {dataset_id} 삭제 완료")
