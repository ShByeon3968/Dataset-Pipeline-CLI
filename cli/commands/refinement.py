"""
pipeline refinement <sub-command>

  errors        레이블 오류 후보 탐지 (클래스 미할당, bbox 이상치)
  filter-bbox   bbox 면적 범위 바깥 어노테이션 삭제
  delete-dupes  중복 이미지 일괄 삭제
  delete-images 이미지 ID 목록으로 일괄 삭제
"""
from __future__ import annotations

from pathlib import Path

import typer

from cli import client, display

app = typer.Typer(help="데이터 정제")


@app.command("errors")
def detect_errors(
    dataset_id: int  = typer.Option(...,   "--dataset-id", "-d", help="데이터셋 ID"),
    out:        Path = typer.Option(None,  "--out",        "-o",
                                    help="결과 JSON 저장 경로 (생략 시 화면 출력만)"),
):
    """
    레이블 오류 후보를 탐지합니다.

    \b
    탐지 항목:
      - 클래스 미할당 어노테이션
      - bbox 경계 초과 (좌표가 0~1 범위 벗어남)
      - bbox 면적 0 (너비 또는 높이 = 0)
      - bbox 면적 이상치 (하위 1% / 상위 99%, 최소 20개 이상일 때)
    """
    try:
        with display.console.status("[bold cyan]오류 탐지 중…"):
            result = client.get(f"/api/v1/datasets/{dataset_id}/refinement/label-errors")
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    total_errors = result.get("total_errors", 0)
    summary      = result.get("issue_summary", {})

    display.print_kv({
        "총 bbox 어노테이션":    result.get("total_bbox_annotations", "-"),
        "오류 후보 수":          total_errors,
        "백분위 탐지 적용":      "예" if result.get("percentile_detection_applied") else "아니오 (20개 미만)",
        **{k: v for k, v in summary.items()},
    }, title=f"레이블 오류 탐지 결과 — 데이터셋 #{dataset_id}")

    errors = result.get("errors", [])
    if errors:
        rows = [
            [e["annotation_id"], e["image_id"], e["image_filename"],
             e["issue"], f"{e['confidence']:.2f}", str(e.get("detail") or "-")]
            for e in errors[:50]
        ]
        display.print_table(
            ["Ann ID", "Img ID", "파일명", "오류 유형", "신뢰도", "상세"],
            rows, title="오류 후보 목록 (최대 50건)"
        )
        if len(errors) > 50:
            display.info(f"… 외 {len(errors) - 50}건 생략. --out 으로 전체 저장 가능.")

    if out:
        import json
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        display.success(f"전체 결과 저장: {out}")

    if total_errors == 0:
        display.success("오류 후보가 없습니다.")


@app.command("filter-bbox")
def filter_bbox(
    dataset_id: int   = typer.Option(...,  "--dataset-id", "-d", help="데이터셋 ID"),
    min_area:   float = typer.Option(0.0,  "--min-area",         help="최소 bbox 면적 (정규화, 0~1)"),
    max_area:   float = typer.Option(1.0,  "--max-area",         help="최대 bbox 면적 (정규화, 0~1)"),
    dry_run:    bool  = typer.Option(True,  "--dry-run/--execute",
                                     help="dry-run: 삭제 대상 건수만 확인 (기본)\nexecute: 실제 삭제 수행"),
    yes:        bool  = typer.Option(False, "--yes", "-y",        help="실행 시 확인 생략"),
):
    """
    bbox 면적 범위 바깥의 어노테이션을 삭제합니다.

    \b
    # 먼저 dry-run으로 확인
    pipeline refinement filter-bbox --dataset-id 1 --min-area 0.001 --max-area 0.8

    \b
    # 실제 삭제
    pipeline refinement filter-bbox --dataset-id 1 --min-area 0.001 --max-area 0.8 --execute
    """
    try:
        result = client.post(
            f"/api/v1/datasets/{dataset_id}/refinement/filter-bbox",
            json={"min_area": min_area, "max_area": max_area, "dry_run": dry_run},
        )
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    to_delete = result.get("to_delete", 0)

    display.print_kv({
        "총 bbox 어노테이션": result.get("total_annotations", "-"),
        "삭제 대상":          to_delete,
        "면적 범위":          f"{min_area} ~ {max_area}",
        "모드":               "dry-run (미리보기)" if result.get("dry_run") else "실제 삭제",
    }, title="bbox 면적 필터")

    if dry_run:
        if to_delete > 0:
            display.warn(f"{to_delete}개 어노테이션이 삭제됩니다.")
            display.info("실제 삭제: --execute 옵션을 추가하세요.")
        else:
            display.success("삭제 대상 어노테이션이 없습니다.")
    else:
        display.success(f"{to_delete}개 어노테이션 삭제 완료")


@app.command("delete-dupes")
def delete_duplicates(
    dataset_id: int  = typer.Option(...,   "--dataset-id", "-d", help="데이터셋 ID"),
    keep:       str  = typer.Option("first", "--keep",
                                    help="중복 그룹에서 유지할 이미지: first | last"),
    mode:       str  = typer.Option("exact", "--mode",
                                    help="중복 기준: exact (MD5) | perceptual (pHash)"),
    yes:        bool = typer.Option(False, "--yes", "-y", help="확인 생략"),
):
    """중복 이미지를 탐지하고 keep 옵션에 따라 하나만 남기고 삭제합니다."""
    try:
        with display.console.status("[bold cyan]중복 탐지 중…"):
            dup_result = client.get(f"/api/v1/datasets/{dataset_id}/analysis/duplicates")
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    groups = dup_result.get(mode, dup_result.get("exact", []))
    if not groups:
        display.success("중복 이미지가 없습니다."); return

    # 삭제 대상 수집
    to_delete: list[int] = []
    for grp in groups:
        imgs = grp["images"]
        imgs_sorted = sorted(imgs, key=lambda x: x["id"])
        keep_img = imgs_sorted[0] if keep == "first" else imgs_sorted[-1]
        to_delete.extend(i["id"] for i in imgs_sorted if i["id"] != keep_img["id"])

    display.info(f"중복 그룹: {len(groups)}개  │  삭제 대상: {len(to_delete)}개 이미지")

    if not yes:
        typer.confirm(f"{len(to_delete)}개 이미지를 삭제하시겠습니까?", abort=True)

    try:
        result = client.post(
            f"/api/v1/datasets/{dataset_id}/images/bulk-delete",
            json={"image_ids": to_delete},
        )
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    display.success(f"{len(to_delete)}개 중복 이미지 삭제 완료")


@app.command("delete-images")
def delete_images(
    dataset_id: int  = typer.Option(...,  "--dataset-id", "-d", help="데이터셋 ID"),
    ids:        str  = typer.Option(...,  "--ids",
                                    help="삭제할 이미지 ID (쉼표 구분). 예: 1,2,3"),
    yes:        bool = typer.Option(False, "--yes", "-y", help="확인 생략"),
):
    """이미지 ID 목록으로 일괄 삭제합니다."""
    id_list = [int(i.strip()) for i in ids.split(",") if i.strip().isdigit()]
    if not id_list:
        display.error("유효한 이미지 ID가 없습니다."); raise typer.Exit(1)

    if not yes:
        typer.confirm(f"{len(id_list)}개 이미지를 삭제하시겠습니까?", abort=True)

    try:
        result = client.post(
            f"/api/v1/datasets/{dataset_id}/images/bulk-delete",
            json={"image_ids": id_list},
        )
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    display.success(f"{len(id_list)}개 이미지 삭제 완료")
