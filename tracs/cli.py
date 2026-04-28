"""Command-line interface for the TraCS agentic Resin code generation pipeline."""

#
# Copyright (c) Simon Kohaut, Felix Divo, Julius Hahnewald, and contributors
#
# This file is part of TraCS and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with TraCS.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

import argparse
import json
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


def _write_trace(output_path: Path, args, regulation: str, iterations: list, accepted: bool, result):
    trace = {
        "parameters": {
            "scenario": Path(args.regulation).stem,
            "natural_language_rule": regulation,
            "system_coder": Path("prompt/system_coder.txt").read_text(encoding="utf-8"),
            "system_judge": Path("prompt/system_judge.txt").read_text(encoding="utf-8"),
            "backend": args.backend,
            "coder_model": args.coder_model,
            "judge_model": args.judge_model,
            "max_iterations": args.max_iterations,
            "seed": args.seed,
            "coder_temperature": args.coder_temperature,
        },
        "iterations": iterations,
        "accepted": accepted,
        "result": result,
    }
    output_path.with_suffix(".json").write_text(json.dumps(trace, indent=2), encoding="utf-8")


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
        "-o", "--output",
        default="-",
        metavar="FILE",
        help="Output path for the accepted .resin program (default: stdout). Also writes a .json trace alongside.",
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
        default="google/gemma-4-E4B-it",
        metavar="MODEL",
        help="Model name for the coder agent (default: google/gemma-4-E4B-it).",
    )
    parser.add_argument(
        "--judge-model",
        default="google/gemma-4-E4B-it",
        metavar="MODEL",
        help="Model name for the judge agent (default: google/gemma-4-E4B-it).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        metavar="N",
        help="Maximum refinement iterations (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="N",
        help="Random seed for reproducibility (not supported by the anthropic backend).",
    )
    parser.add_argument(
        "--coder-temperature",
        type=float,
        default=0.3,
        metavar="T",
        help="Sampling temperature for the coder agent (default: 0.3).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help=f"Show the full coder output at each iteration (default: first {_TRUNCATE_LINES} lines).",
    )
    args = parser.parse_args()

    console = Console(highlight=False)
    iterations = []

    try:
        try:
            regulation = Path(args.regulation).read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)

        console.rule("[bold cyan]TraCS[/bold cyan]")
        console.print(f"  Regulation : [cyan]{args.regulation}[/cyan]")
        console.print(f"  Backend    : [cyan]{args.backend}[/cyan]")
        console.print(f"  Coder      : [cyan]{args.coder_model}[/cyan]")
        console.print(f"  Judge      : [cyan]{args.judge_model}[/cyan]")
        console.print(f"  Max iter.  : [cyan]{args.max_iterations}[/cyan]")
        console.print()

        pipeline = CoderPipeline(
            coder_model=args.coder_model,
            judge_model=args.judge_model,
            backend=args.backend,
            max_iterations=args.max_iterations,
            seed=args.seed,
            coder_temperature=args.coder_temperature,
        )

        def on_iteration(record: IterationRecord) -> None:
            console.print(_iteration_panel(record, args.max_iterations, args.verbose))
            iterations.append({
                "program": record.program,
                "judge_passed": record.judge_passed,
                "critique": record.critique,
                "compiled": record.compiled,
                "compiler_errors": record.diagnostics,
            })

        program, history = pipeline.run(regulation, on_iteration=on_iteration)
        n = len(history)

        console.rule(f"[green bold]Accepted after {n} iteration{'s' if n != 1 else ''}[/green bold]")

        if args.output == "-":
            console.print(Syntax(program, "prolog", theme="monokai", line_numbers=True))
        else:
            output_path = Path(args.output)
            output_path.write_text(program, encoding="utf-8")
            _write_trace(output_path, args, regulation, iterations, accepted=True, result=program)
            console.print(f"  Saved to: [green]{args.output}[/green]")

    except RuntimeError as exc:
        console.rule("[red bold]Pipeline Failed[/red bold]")
        console.print(f"  [red]{exc}[/red]")
        if args.output != "-":
            _write_trace(Path(args.output), args, regulation, iterations, accepted=False, result=None)
        sys.exit(1)


if __name__ == "__main__":
    main()
