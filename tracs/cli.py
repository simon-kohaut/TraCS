"""Command-line interface for the TraCS agentic Resin code generation pipeline."""

#
# Copyright (c) Simon Kohaut, Felix Divo, Julius Hahnewald, and contributors
#
# This file is part of TraCS and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with TraCS.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from tracs.coder import CoderPipeline, IterationRecord

_TRUNCATE_LINES = 15


def _program_renderable(program: str, verbose: bool):
    lines = program.splitlines()
    if verbose or len(lines) <= _TRUNCATE_LINES:
        return Syntax(program, "prolog", theme="ansi_dark")
    truncated = "\n".join(lines[:_TRUNCATE_LINES])
    return Group(
        Syntax(truncated, "prolog", theme="ansi_dark"),
        Text(f"  … {len(lines) - _TRUNCATE_LINES} more line(s) — use -v to show all", style="dim italic"),
    )


def _iteration_panel(record: IterationRecord, max_iterations: int, verbose: bool) -> Panel:
    content = [_program_renderable(record.program, verbose), Text("")]

    if record.judge_passed:
        content.append(Text("  Judge    ✓ PASS", style="green bold"))
    else:
        content.append(Text("  Judge    ✗ FAIL", style="red bold"))
        for line in record.critique.splitlines():
            content.append(Text(f"    {line}", style="red"))

    content.append(Text(""))

    if record.compiled:
        content.append(Text("  Compiler ✓ OK", style="green bold"))
    else:
        content.append(Text("  Compiler ✗ ERROR", style="red bold"))
        for line in record.diagnostics.splitlines():
            content.append(Text(f"    {line}", style="red"))

    accepted = record.judge_passed and record.compiled
    return Panel(
        Group(*content),
        title=f"[bold]Iteration {record.iteration} / {max_iterations}[/bold]",
        border_style="green" if accepted else "red",
        box=box.ROUNDED,
    )


def _log_iteration(record: IterationRecord, max_iterations: int) -> str:
    sep = "-" * 60
    lines = [
        "=" * 60,
        f"Iteration {record.iteration} / {max_iterations}",
        "=" * 60,
        "",
        "Coder output:",
        sep,
        record.program,
        sep,
        "",
        f"Judge:    {'PASS' if record.judge_passed else 'FAIL'}",
    ]
    if not record.judge_passed:
        lines.append(f"Critique: {record.critique}")
    lines += ["", f"Compiler: {'OK' if record.compiled else 'ERROR'}"]
    if not record.compiled:
        lines += ["Errors:", record.diagnostics]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="tracs",
        description="Generate a Resin compliance model from a natural-language traffic regulation.",
    )
    parser.add_argument(
        "regulation",
        help="Text file containing the natural-language regulation description.",
    )
    parser.add_argument(
        "-l", "--library",
        required=True,
        metavar="FILE",
        help="Resin spatial relation library (.txt file).",
    )
    parser.add_argument(
        "-o", "--output",
        default="-",
        metavar="FILE",
        help="Output path for the accepted program (default: stdout).",
    )
    parser.add_argument(
        "--log",
        metavar="FILE",
        help="Write a plain-text session log to this file.",
    )
    parser.add_argument(
        "--backend",
        default="ollama",
        choices=["ollama", "openai", "anthropic", "local"],
        help=(
            "LLM backend to use (default: ollama). "
            "'openai' reads OPENAI_API_KEY, 'anthropic' reads ANTHROPIC_API_KEY, "
            "'local' reads LOCAL_MODEL_BASE_URL and LOCAL_MODEL_KEY from .env."
        ),
    )
    parser.add_argument(
        "--coder-model",
        default="qwen2.5:7b-instruct",
        metavar="MODEL",
        help="Model name for the coder agent (default: qwen2.5:7b-instruct).",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen2.5:7b-instruct",
        metavar="MODEL",
        help="Model name for the judge agent (default: qwen2.5:7b-instruct).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        metavar="N",
        help="Maximum refinement iterations (default: 10).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help=f"Show the full coder output at each iteration (default: first {_TRUNCATE_LINES} lines).",
    )
    args = parser.parse_args()

    console = Console(highlight=False)
    log_file = open(args.log, "w", encoding="utf-8") if args.log else None

    try:
        try:
            regulation = Path(args.regulation).read_text(encoding="utf-8").strip()
            library = Path(args.library).read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)

        console.rule("[bold cyan]TraCS[/bold cyan]")
        console.print(f"  Regulation : [cyan]{args.regulation}[/cyan]")
        console.print(f"  Library    : [cyan]{args.library}[/cyan]")
        console.print(f"  Backend    : [cyan]{args.backend}[/cyan]")
        console.print(f"  Coder      : [cyan]{args.coder_model}[/cyan]")
        console.print(f"  Judge      : [cyan]{args.judge_model}[/cyan]")
        console.print(f"  Max iter.  : [cyan]{args.max_iterations}[/cyan]")
        console.print()

        if log_file:
            log_file.write(
                f"TraCS session\n"
                f"Regulation : {args.regulation}\n"
                f"Library    : {args.library}\n"
                f"Backend    : {args.backend}\n"
                f"Coder      : {args.coder_model}\n"
                f"Judge      : {args.judge_model}\n"
                f"Max iter.  : {args.max_iterations}\n\n"
            )

        pipeline = CoderPipeline(
            coder_model=args.coder_model,
            judge_model=args.judge_model,
            backend=args.backend,
            max_iterations=args.max_iterations,
        )

        def on_iteration(record: IterationRecord) -> None:
            console.print(_iteration_panel(record, args.max_iterations, args.verbose))
            if log_file:
                log_file.write(_log_iteration(record, args.max_iterations))
                log_file.flush()

        program, history = pipeline.run(regulation, library, on_iteration=on_iteration)
        n = len(history)

        console.rule(f"[green bold]Accepted after {n} iteration{'s' if n != 1 else ''}[/green bold]")
        console.print()

        if args.output == "-":
            console.print(Syntax(program, "prolog", theme="monokai", line_numbers=True))
        else:
            Path(args.output).write_text(program, encoding="utf-8")
            console.print(f"  Saved to: [green]{args.output}[/green]")

        if log_file:
            log_file.write(
                f"\n{'=' * 60}\n"
                f"Accepted after {n} iteration(s)\n"
                f"{'=' * 60}\n\n"
                f"{program}\n"
            )

    except RuntimeError as exc:
        console.rule("[red bold]Pipeline Failed[/red bold]")
        console.print(f"  [red]{exc}[/red]")
        if log_file:
            log_file.write(f"\nPipeline failed: {exc}\n")
        sys.exit(1)
    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
