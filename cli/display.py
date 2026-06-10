"""
터미널 출력 헬퍼 (rich 기반)

사용법:
  from cli.display import console, table, success, error, info, warn
"""
from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()
err_console = Console(stderr=True)


# ── 메시지 출력 ───────────────────────────────────────────────────────

def success(msg: str) -> None:
    console.print(f"[bold green]✓[/bold green]  {msg}")


def error(msg: str) -> None:
    err_console.print(f"[bold red]✗[/bold red]  {msg}")


def info(msg: str) -> None:
    console.print(f"[bold cyan]ℹ[/bold cyan]  {msg}")


def warn(msg: str) -> None:
    console.print(f"[bold yellow]⚠[/bold yellow]  {msg}")


# ── 테이블 빌더 ───────────────────────────────────────────────────────

def make_table(columns: list[str], rows: list[list[Any]],
               title: str | None = None) -> Table:
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta",
              title=title, title_style="bold")
    for col in columns:
        t.add_column(col, overflow="fold")
    for row in rows:
        t.add_row(*[str(v) if v is not None else "-" for v in row])
    return t


def print_table(columns: list[str], rows: list[list[Any]],
                title: str | None = None) -> None:
    console.print(make_table(columns, rows, title))


def print_kv(data: dict, title: str | None = None) -> None:
    """key-value 쌍을 2열 테이블로 출력"""
    t = Table(box=box.SIMPLE, show_header=False)
    t.add_column("Key",   style="bold cyan",  min_width=20)
    t.add_column("Value", style="white")
    for k, v in data.items():
        t.add_row(str(k), str(v) if v is not None else "-")
    if title:
        console.print(Panel(t, title=title, border_style="cyan"))
    else:
        console.print(t)


def print_json(data: Any) -> None:
    from rich.pretty import Pretty
    console.print(Pretty(data))
