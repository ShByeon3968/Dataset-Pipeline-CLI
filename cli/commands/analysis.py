"""
pipeline analysis <sub-command>

  stats         클래스 분포, 이미지 수, 어노테이션 수 통계
  duplicates    중복 이미지 탐지 (퍼셉추얼 해시 + MD5)
  embeddings    이미지 임베딩 계산 및 UMAP 좌표 CSV 저장
  errors        레이블 오류 후보 탐지
"""
from __future__ import annotations

from pathlib import Path

import typer

from cli import client, display

app = typer.Typer(help="데이터셋 분석")


@app.command("stats")
def stats(
    dataset_id: int = typer.Option(..., "--dataset-id", "-d", help="데이터셋 ID"),
):
    """클래스 분포, 이미지/어노테이션 수 등 통계를 조회합니다."""
    try:
        result = client.get(f"/api/v1/datasets/{dataset_id}/analysis/stats")
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    # 개요
    display.print_kv({
        "데이터셋 ID":     dataset_id,
        "총 이미지":       result.get("total_images", "-"),
        "총 어노테이션":   result.get("total_annotations", "-"),
        "클래스 수":       result.get("total_classes", "-"),
        "어노테이션 없는 이미지": result.get("unannotated_images", "-"),
    }, title="데이터셋 통계")

    # 클래스별 분포
    dist = result.get("class_distribution") or result.get("class_stats", [])
    if dist:
        rows = [[c.get("name", c.get("class_name")),
                 c.get("count", c.get("annotation_count", 0))]
                for c in dist]
        rows.sort(key=lambda x: -int(x[1]))
        display.print_table(["클래스", "어노테이션 수"], rows, title="클래스 분포")


@app.command("duplicates")
def duplicates(
    dataset_id: int = typer.Option(..., "--dataset-id", "-d", help="데이터셋 ID"),
):
    """중복 이미지를 탐지합니다 (퍼셉추얼 해시 / MD5)."""
    try:
        with display.console.status("[bold cyan]중복 탐지 중…"):
            result = client.get(f"/api/v1/datasets/{dataset_id}/analysis/duplicates")
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    perceptual = result.get("perceptual", [])
    exact      = result.get("exact", [])

    display.print_kv({
        "퍼셉추얼 해시 중복 그룹": len(perceptual),
        "완전 중복(MD5) 그룹":     len(exact),
    }, title="중복 탐지 결과")

    if exact:
        display.info("[bold]완전 중복 그룹 (MD5)[/bold]")
        for grp in exact[:10]:
            names = ", ".join(i["filename"] for i in grp["images"])
            display.console.print(f"  hash={grp['file_hash'][:12]}…  ({grp['count']}개): {names}")
        if len(exact) > 10:
            display.info(f"  … 외 {len(exact) - 10}개 그룹 생략")

    if perceptual:
        display.info("[bold]유사 이미지 그룹 (pHash)[/bold]")
        for grp in perceptual[:10]:
            names = ", ".join(i["filename"] for i in grp["images"])
            display.console.print(f"  phash={grp['phash'][:12]}…  ({grp['count']}개): {names}")
        if len(perceptual) > 10:
            display.info(f"  … 외 {len(perceptual) - 10}개 그룹 생략")

    if not perceptual and not exact:
        display.success("중복 이미지가 없습니다.")


@app.command("embeddings")
def embeddings(
    dataset_id: int  = typer.Option(...,  "--dataset-id", "-d", help="데이터셋 ID"),
    method:     str  = typer.Option("umap", "--method", "-m",   help="시각화 방법: umap | tsne | pca"),
    out:        Path = typer.Option(Path("."), "--out", "-o",   help="CSV 저장 디렉토리"),
):
    """이미지 임베딩을 계산하고 2D 좌표를 CSV로 저장합니다."""
    try:
        with display.console.status("[bold cyan]임베딩 계산 중… (수백 이미지면 수 분 소요)"):
            result = client.get(
                f"/api/v1/datasets/{dataset_id}/analysis/embeddings",
                params={"method": method},
            )
    except client.PipelineError as e:
        display.error(str(e)); raise typer.Exit(1)

    points = result.get("points", result if isinstance(result, list) else [])
    if not points:
        display.warn("임베딩 데이터가 없습니다."); return

    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"embeddings_{dataset_id}_{method}.csv"

    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "filename", "x", "y",
                         "class_names", "annotation_count"])
        for p in points:
            writer.writerow([
                p.get("image_id"), p.get("filename"),
                p.get("x"), p.get("y"),
                "|".join(p.get("class_names", [])),
                p.get("annotation_count", 0),
            ])

    display.success(f"{len(points)}개 이미지 임베딩 저장: [bold]{csv_path}[/bold]")
    display.info(f"방법: {method}  │  모델: {result.get('model', '-')}")
