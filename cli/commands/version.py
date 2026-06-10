"""
pipeline version <sub-command>

  create    데이터셋 버전 스냅샷 생성
  list      버전 목록 조회
  info      버전 상세 (클래스 분포 포함)
  delete    버전 삭제
  diff      두 버전 간 차이 비교
  lineage   데이터셋↔모델 리니지 그래프 출력

  model-create   모델 버전 등록
  model-list     모델 버전 목록
  model-link     모델 버전 ↔ 데이터셋 버전 연결
"""
from __future__ import annotations

import json

import typer

from cli import client, display

app = typer.Typer(help="버저닝 및 모델 리니지")


# ── 데이터셋 버전 ─────────────────────────────────────────────────────

@app.command("create")
def create_version(
    dataset_id:  int  = typer.Option(..., "--dataset-id",  "-d", help="데이터셋 ID"),
    name:        str  = typer.Option(..., "--name",        "-n", help="버전 이름. 예: v1.0"),
    desc:        str  = typer.Option("",  "--desc",               help="설명"),
    branch:      str  = typer.Option("main", "--branch",          help="브랜치 이름"),
    parent_id:   int | None = typer.Option(None, "--parent-id",   help="부모 버전 ID"),
    tags:        str  = typer.Option("",  "--tags",               help="태그 (쉼표 구분)"),
    created_by:  str  = typer.Option("user", "--created-by",      help="작성자"),
):
    """현재 데이터셋 상태의 버전 스냅샷을 생성합니다."""
    try:
        result = client.post(
            f"/api/v1/datasets/{dataset_id}/versions",
            json={
                "version_name":    name,
                "description":     desc,
                "branch_name":     branch,
                "parent_version_id": parent_id,
                "tags":            tags,
                "created_by":      created_by,
            },
        )
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    display.success(f"버전 생성: [bold]{result['version_name']}[/bold]  (id={result['id']})")
    display.print_kv({
        "버전 ID":      result["id"],
        "이름":         result["version_name"],
        "브랜치":       result.get("branch_name"),
        "이미지 수":    result.get("image_count"),
        "어노테이션 수": result.get("annotation_count"),
        "클래스 수":    result.get("class_count"),
        "추가 이미지":  result.get("added_images"),
        "삭제 이미지":  result.get("deleted_images"),
        "변경 레이블":  result.get("modified_labels"),
        "태그":         result.get("tags") or "-",
        "생성일":       result.get("created_at", "-")[:19] if result.get("created_at") else "-",
    }, title="생성된 버전")


@app.command("list")
def list_versions(
    dataset_id: int        = typer.Option(...,  "--dataset-id", "-d", help="데이터셋 ID"),
    branch:     str | None = typer.Option(None, "--branch",           help="브랜치 필터"),
    skip:       int        = typer.Option(0,    "--skip"),
    limit:      int        = typer.Option(50,   "--limit"),
):
    """데이터셋 버전 목록을 조회합니다."""
    params: dict = {"skip": skip, "limit": limit}
    if branch:
        params["branch"] = branch
    try:
        result = client.get(f"/api/v1/datasets/{dataset_id}/versions", params=params)
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    items = result.get("items", result if isinstance(result, list) else [])
    if not items:
        display.info("버전이 없습니다."); return

    rows = [
        [v["id"], v["version_name"], v.get("branch_name"),
         v.get("image_count"), v.get("annotation_count"),
         f"+{v.get('added_images',0)} / -{v.get('deleted_images',0)}",
         v.get("tags") or "-",
         v.get("created_at", "-")[:10] if v.get("created_at") else "-"]
        for v in items
    ]
    display.print_table(
        ["ID", "이름", "브랜치", "이미지", "어노테이션", "변경", "태그", "생성일"],
        rows, title=f"데이터셋 #{dataset_id} 버전 목록"
    )


@app.command("info")
def version_info(
    version_id: int = typer.Option(..., "--version-id", "-v", help="버전 ID"),
):
    """버전 상세 정보 및 클래스 분포를 조회합니다."""
    try:
        v = client.get(f"/api/v1/versions/{version_id}")
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    display.print_kv({
        "버전 ID":       v["id"],
        "이름":          v["version_name"],
        "브랜치":        v.get("branch_name"),
        "설명":          v.get("description") or "-",
        "이미지 수":     v.get("image_count"),
        "어노테이션 수":  v.get("annotation_count"),
        "클래스 수":     v.get("class_count"),
        "추가 이미지":   v.get("added_images"),
        "삭제 이미지":   v.get("deleted_images"),
        "변경 레이블":   v.get("modified_labels"),
        "이미지 해시":   v.get("image_ids_hash", "-")[:16] + "…",
        "태그":          v.get("tags") or "-",
        "생성자":        v.get("created_by"),
        "생성일":        v.get("created_at", "-")[:19] if v.get("created_at") else "-",
    }, title=f"버전 #{version_id}")

    # 클래스 분포
    dist_raw = v.get("class_distribution", "[]")
    try:
        dist = json.loads(dist_raw) if isinstance(dist_raw, str) else dist_raw
    except Exception:
        dist = []

    if dist:
        rows = [[d["name"], d["count"]] for d in dist]
        rows.sort(key=lambda x: -x[1])
        display.print_table(["클래스", "어노테이션 수"], rows, title="클래스 분포")


@app.command("delete")
def delete_version(
    version_id: int  = typer.Option(...,  "--version-id", "-v", help="버전 ID"),
    yes:        bool = typer.Option(False, "--yes", "-y",        help="확인 생략"),
):
    """버전을 삭제합니다."""
    if not yes:
        typer.confirm(f"버전 {version_id}을(를) 삭제하시겠습니까?", abort=True)
    try:
        client.delete(f"/api/v1/versions/{version_id}")
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)
    display.success(f"버전 {version_id} 삭제 완료")


@app.command("lineage")
def show_lineage(
    dataset_id: int = typer.Option(..., "--dataset-id", "-d", help="데이터셋 ID"),
):
    """데이터셋과 연결된 모델 버전의 리니지 그래프를 출력합니다."""
    try:
        result = client.get(f"/api/v1/datasets/{dataset_id}/lineage")
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    nodes = result.get("nodes", [])
    edges = result.get("edges", [])

    if not nodes:
        display.info("리니지 데이터가 없습니다."); return

    display.console.print(f"\n[bold]리니지 그래프 — 데이터셋 #{dataset_id}[/bold]")
    display.console.print(f"  노드: {len(nodes)}개   엣지: {len(edges)}개\n")

    # 노드 테이블
    rows = [
        [n["id"], n["type"], n["label"],
         n.get("created_at", "-")[:10] if n.get("created_at") else "-"]
        for n in nodes
    ]
    display.print_table(["ID", "타입", "레이블", "생성일"], rows, title="노드")

    # 엣지 테이블
    if edges:
        edge_rows = [
            [e["source"], e["source_type"], "→", e["target"], e["target_type"], e.get("label", "-")]
            for e in edges
        ]
        display.print_table(
            ["소스 ID", "소스 타입", "", "대상 ID", "대상 타입", "레이블"],
            edge_rows, title="엣지"
        )


# ── 모델 버전 ─────────────────────────────────────────────────────────

@app.command("model-create")
def model_create(
    name:       str        = typer.Option(...,  "--name",      "-n",  help="모델 이름"),
    framework:  str        = typer.Option("",   "--framework",        help="프레임워크. 예: YOLOv8"),
    desc:       str        = typer.Option("",   "--desc",             help="설명"),
    created_by: str        = typer.Option("user", "--created-by"),
):
    """모델 버전을 등록합니다."""
    try:
        result = client.post("/api/v1/model-versions", json={
            "name": name, "framework": framework,
            "description": desc, "created_by": created_by,
        })
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    display.success(f"모델 버전 등록: [bold]{result['name']}[/bold]  (id={result['id']})")


@app.command("model-list")
def model_list(
    skip:  int = typer.Option(0,  "--skip"),
    limit: int = typer.Option(50, "--limit"),
):
    """등록된 모델 버전 목록을 조회합니다."""
    try:
        result = client.get("/api/v1/model-versions", params={"skip": skip, "limit": limit})
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    items = result.get("items", result if isinstance(result, list) else [])
    if not items:
        display.info("등록된 모델 버전이 없습니다."); return

    rows = [
        [m["id"], m["name"], m.get("framework") or "-", m.get("description") or "-",
         m.get("created_at", "-")[:10] if m.get("created_at") else "-"]
        for m in items
    ]
    display.print_table(["ID", "이름", "프레임워크", "설명", "등록일"], rows,
                        title="모델 버전 목록")


@app.command("model-link")
def model_link(
    model_version_id:   int = typer.Option(..., "--model-id",   "-m", help="모델 버전 ID"),
    dataset_version_id: int = typer.Option(..., "--version-id", "-v", help="데이터셋 버전 ID"),
    dataset_id:         int = typer.Option(..., "--dataset-id", "-d", help="데이터셋 ID"),
    note:               str = typer.Option("",  "--note",             help="메모"),
    linked_by:          str = typer.Option("user", "--linked-by"),
):
    """모델 버전과 데이터셋 버전을 연결합니다."""
    try:
        result = client.post(
            f"/api/v1/model-versions/{model_version_id}/links",
            json={
                "dataset_version_id": dataset_version_id,
                "dataset_id":         dataset_id,
                "note":               note,
                "linked_by":          linked_by,
            },
        )
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    display.success(
        f"연결 완료: 모델 v{model_version_id} ↔ 데이터셋 버전 {dataset_version_id}  "
        f"(link_id={result['id']})"
    )
