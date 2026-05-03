"""CLI script to pretty-print a TraCS JSON trace file."""

import argparse
import json
import sys
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


_PROMPT_TRUNCATE_LINES = 5


def _truncated_text(text: str, n: int) -> list:
    lines = text.splitlines()
    items = [Text(line) for line in lines[:n]]
    if len(lines) > n:
        items.append(Text(f"  … {len(lines) - n} more line(s)", style="dim italic"))
    return items


def _program_renderable(program: str):
    return Syntax(program, "prolog", theme="ansi_dark", word_wrap=True)


def _iteration_panel(idx: int, record: dict, n_total: int) -> Panel:
    content = [_program_renderable(record["program"]), Text("")]

    if record["judge_passed"]:
        content.append(Text("  Judge    ✓ PASS", style="green bold"))
    else:
        content.append(Text("  Judge    ✗ FAIL", style="red bold"))
        for line in record["critique"].splitlines():
            content.append(Text(f"    {line}", style="red"))

    content.append(Text(""))

    compiler_errors = record.get("compiler_errors", "")
    if record["compiled"]:
        content.append(Text("  Compiler ✓ PASS", style="green bold"))
    else:
        content.append(Text("  Compiler ✗ FAIL", style="red bold"))
        for line in compiler_errors.splitlines():
            content.append(Text(f"    {line}", style="red"))

    accepted = record["judge_passed"] and record["compiled"]
    return Panel(
        Group(*content),
        title=f"[bold]Iteration {idx} / {n_total}[/bold]",
        border_style="green" if accepted else "red",
        box=box.ROUNDED,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tracs_show",
        description="Pretty-print a TraCS JSON trace file.",
    )
    parser.add_argument("trace", help="Path to the .json trace file.")
    args = parser.parse_args()

    console = Console(highlight=False)

    try:
        trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Error:[/red] Invalid JSON — {exc}")
        sys.exit(1)

    params = trace["parameters"]
    iterations = trace["iterations"]
    n = len(iterations)

    console.rule("[bold cyan]TraCS Trace[/bold cyan]")

    def _param(label: str, value) -> Text:
        return Text.from_markup(f"  [bold]{label:<14}[/bold] [cyan]{value}[/cyan]")

    def _heading(label: str) -> Text:
        return Text.from_markup(f"\n  [bold underline]{label}[/bold underline]")

    param_content = [
        _param("Scenario", params["scenario"]),
        _param("Backend", params["backend"]),
        _param("Coder model", params["coder_model"]),
        _param("Judge model", params["judge_model"]),
        _param("Max iter.", params["max_iterations"]),
        _param("Seed", params["seed"]),
        _param("Coder temp.", params["coder_temperature"]),
        _heading("Regulation"),
        *[Text(f"  {line}") for line in params["natural_language_rule"].splitlines()],
        _heading("System coder"),
        *_truncated_text(params["system_coder"], _PROMPT_TRUNCATE_LINES),
        _heading("System judge"),
        *_truncated_text(params["system_judge"], _PROMPT_TRUNCATE_LINES),
    ]
    console.print(Panel(Group(*param_content), title="[bold]Parameters[/bold]", box=box.ROUNDED))

    for i, record in enumerate(iterations, start=1):
        console.print(_iteration_panel(i, record, n))

    if trace["accepted"]:
        console.rule("[green bold]Accepted[/green bold]")
        console.print(Panel(_program_renderable(trace["full_result"]), box=box.ROUNDED))
    else:
        console.rule("[red bold]Not Accepted[/red bold]")


if __name__ == "__main__":
    main()
