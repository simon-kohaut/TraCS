"""CLI script to post-hoc add judge opinions to a TraCS JSON trace file."""

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
from os import getenv
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

from tracs.coder import JudgeOutput


def _build_judge_llm(backend: str, model: str, seed: int):
    match backend:
        case "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=0.0, seed=seed).with_structured_output(JudgeOutput)
        case "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, max_tokens=8_000).with_structured_output(JudgeOutput)
        case "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, temperature=0.0, seed=seed).with_structured_output(JudgeOutput)
        case "local":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                base_url=getenv("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:54112/v1"),
                api_key=getenv("LOCAL_MODEL_KEY", ""),
                temperature=0.0,
                seed=seed,
            ).with_structured_output(JudgeOutput)
        case "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model, temperature=0.0, seed=seed).with_structured_output(JudgeOutput)
        case _:
            raise ValueError(f"Unknown backend: {backend!r}")


def _migrate_iteration(it: dict) -> None:
    """Migrate old human_judge_passed field to the opinions dict."""
    if "opinions" not in it:
        it["opinions"] = {}
    if "human_judge_passed" in it:
        val = it.pop("human_judge_passed")
        if val is not None:
            it["opinions"].setdefault("human", {"passed": val})


def _save(trace_path: Path, trace: dict) -> None:
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")


def _run_human(
    console: Console,
    trace_path: Path,
    trace: dict,
    iterations: list,
    judge: str,
    n: int,
) -> None:
    for i, it in enumerate(iterations, start=1):
        program = it.get("program", "")
        console.print(Panel(
            Syntax(program, "prolog", theme="ansi_dark", word_wrap=True),
            title=f"[bold]Iteration {i} / {n}[/bold]",
            box=box.ROUNDED,
        ))

        if judge in it["opinions"]:
            console.print(f"  [dim]Skipping iteration {i}: already has a '{judge}' opinion.[/dim]")
            continue

        answer = Prompt.ask("  Pass?", choices=["y", "n", "s"], default="s")

        if answer == "s":
            console.print("  [dim]Skipped.[/dim]")
            continue

        it["opinions"][judge] = {"passed": answer == "y"}
        _save(trace_path, trace)
        console.print("  [green]Saved.[/green]")

    console.rule("[green bold]Done[/green bold]")


def _run_llm(
    console: Console,
    trace_path: Path,
    trace: dict,
    iterations: list,
    args: argparse.Namespace,
    regulation: str,
    system_judge: str,
    n: int,
) -> None:
    try:
        llm = _build_judge_llm(args.backend, args.judge, args.seed)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    for i, it in enumerate(iterations, start=1):
        if args.judge in it["opinions"]:
            console.print(f"  [yellow]Warning:[/yellow] Overwriting existing opinion for '{args.judge}' on iteration {i}.")

        console.print(f"  Judging iteration {i}/{n} …", end=" ")

        messages = [
            SystemMessage(content=system_judge),
            HumanMessage(content=(
                f"Traffic regulation (natural language):\n{regulation}"
                "\n\n"
                f"Candidate Resin program:\n{it.get('program', '')}"
            )),
        ]
        result: JudgeOutput = llm.invoke(messages)
        it["opinions"][args.judge] = {"passed": result.verdict, "critique": result.critique}
        _save(trace_path, trace)

        status = "[green]PASS[/green]" if result.verdict else "[red]FAIL[/red]"
        console.print(status)

    console.rule("[green bold]Done[/green bold]")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="tracs_add_judge",
        description="Add post-hoc judge opinions to a TraCS JSON trace file.",
    )
    parser.add_argument("trace", help="Path to the .json trace file.")
    parser.add_argument(
        "--judge",
        default="human",
        metavar="JUDGE",
        help=(
            "Judge identifier. Use 'human' for interactive annotation, "
            "or a model name (e.g. claude-opus-4-7) for LLM judging (default: human)."
        ),
    )
    parser.add_argument(
        "--backend",
        default="anthropic",
        choices=["ollama", "openai", "anthropic", "google", "local"],
        help="LLM backend to use (ignored when --judge human, default: anthropic).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="N",
        help="Random seed for the LLM judge (default: 1).",
    )
    args = parser.parse_args()

    console = Console(highlight=False)
    trace_path = Path(args.trace)

    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Error:[/red] Invalid JSON — {exc}")
        sys.exit(1)

    regulation = trace["parameters"]["natural_language_rule"]
    system_judge = trace["parameters"]["system_judge"]
    iterations = trace["iterations"]
    n = len(iterations)

    for it in iterations:
        _migrate_iteration(it)

    console.rule("[bold cyan]TraCS Add Judge[/bold cyan]")
    console.print(f"  Trace      : [cyan]{args.trace}[/cyan]")
    console.print(f"  Judge      : [cyan]{args.judge}[/cyan]")
    console.print(f"  Iterations : [cyan]{n}[/cyan]")
    console.print()
    console.print(Panel(
        "\n".join(f"  {line}" for line in regulation.splitlines()),
        title="[bold]Regulation[/bold]",
        box=box.ROUNDED,
    ))
    console.print()

    if args.judge == "human":
        _run_human(console, trace_path, trace, iterations, args.judge, n)
    else:
        _run_llm(console, trace_path, trace, iterations, args, regulation, system_judge, n)


if __name__ == "__main__":
    main()
