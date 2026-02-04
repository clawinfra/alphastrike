"""Output formatting utilities for the CLI."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_json(data: dict[str, Any]) -> None:
    """Print data as formatted JSON."""
    console.print_json(json.dumps(data, indent=2, default=str))


def print_table(
    title: str,
    columns: list[str],
    rows: list[list[str]],
    show_header: bool = True,
) -> None:
    """Print a formatted table."""
    table = Table(title=title, show_header=show_header)
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*row)
    console.print(table)


def print_banner(
    title: str,
    subtitle: str | None = None,
    items: dict[str, str] | None = None,
) -> None:
    """Print a styled banner/panel."""
    content = ""
    if subtitle:
        content += f"[dim]{subtitle}[/dim]\n\n"
    if items:
        max_key_len = max(len(k) for k in items.keys())
        for key, value in items.items():
            content += f"[cyan]{key:<{max_key_len}}[/cyan]  {value}\n"

    panel = Panel(
        content.strip(),
        title=f"[bold]{title}[/bold]",
        border_style="blue",
    )
    console.print(panel)


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[red bold]Error:[/red bold] {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green bold]Success:[/green bold] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow bold]Warning:[/yellow bold] {message}")


def print_status_box(
    title: str,
    sections: list[dict[str, Any]],
) -> None:
    """Print a status box with multiple sections."""
    lines = []
    for section in sections:
        if section.get("header"):
            lines.append(f"[bold cyan]{section['header']}[/bold cyan]")
        for key, value in section.get("items", {}).items():
            lines.append(f"  [dim]{key}:[/dim] {value}")
        lines.append("")

    panel = Panel(
        "\n".join(lines).strip(),
        title=f"[bold]{title}[/bold]",
        border_style="green",
    )
    console.print(panel)
