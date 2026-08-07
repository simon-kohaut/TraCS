"""Entry point for running all scenarios across all configured models and seeds."""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

# (backend, model)
MODELS = [
    # ("local",      "google/gemma-4-31B-it"),
    # ("local",      "Qwen/Qwen3.6-27B"),
    # ("local",      "Aleph-Alpha/Pharia-1-LLM-7B-control-hf"),
    ("openai",     "gpt-5.5"),
    ("openai",     "gpt-5.4-mini"),
    ("anthropic",  "claude-opus-4-7"),
    ("anthropic",  "claude-sonnet-4-6"),
    ("google",     "gemini-3.1-pro-preview"),
    ("google",     "gemini-3-flash-preview"),
    #
    # ("local",      "Qwen/Qwen3.5-0.8B"),
    # ("local",      "Qwen/Qwen3.5-2B"),
    # ("local",      "Qwen/Qwen3.5-4B"),
    # ("local",      "Qwen/Qwen3.5-9B"),
    # ("local",      "Qwen/Qwen3.5-27B"),
    # ("local",      "Qwen/Qwen3.5-122B-A10B"),
]


def _run_one(scenario: Path, backend: str, model: str, seed: int, out_path: Path, *, capture: bool) -> tuple[bool, str]:
    result = subprocess.run([
        "tracs_agents_run", str(scenario),
        "--output", str(out_path),
        "--coder-backend", backend,
        "--judge-backend", backend,
        "--coder-model", model,
        "--judge-model", model,
        "--seed", str(seed),
        "--coder-temperature", "0.3",
        "--max-iterations", "10",
    ], check=False,
    stdout=subprocess.PIPE if capture else None,
    stderr=subprocess.STDOUT if capture else None,
    text=True)
    output = result.stdout or ""
    return result.returncode == 0, output


def _report(console: Console, out_path: Path, ok: bool, output: str) -> None:
    if ok:
        console.print(f"[green]\\[ok][/green]   {out_path}")
    else:
        console.print(f"[red]\\[fail][/red] {out_path}")
        if output.strip():
            for line in output.splitlines():
                console.print(f"  {line}")


def main():
    parser = argparse.ArgumentParser(prog="tracs_agents_run_all")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel worker threads. Omit for sequential mode with full output.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output"),
        metavar="DIR",
        help="Directory to write result files to (default: output).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        metavar="N",
        help="Number of seeds to run per scenario/model, starting at 1 (default: 3).",
    )
    args = parser.parse_args()

    seeds = range(1, args.seeds + 1)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    console = Console(highlight=False)

    jobs = []
    for scenario in sorted(Path("scenarios").glob("*.txt")):
        for backend, model in MODELS:
            model_slug = model.replace("/", "_")
            for seed in seeds:
                stem = f"{scenario.stem}_{model_slug}_seed{seed}"
                out_path = out / f"{stem}.json"
                if out_path.exists():
                    console.print(f"[dim]\\[skip][/dim] {out_path}")
                else:
                    jobs.append((scenario, backend, model, seed, out_path))

    if not jobs:
        console.print("Nothing to run.")
        return

    n_ok = n_fail = 0

    if args.workers is None:
        for job in jobs:
            *_, out_path = job
            ok, output = _run_one(*job, capture=False)
            # In sequential mode the subprocess output is already visible; just print the status line.
            _report(console, out_path, ok, "")
            if ok:
                n_ok += 1
            else:
                n_fail += 1
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running {len(jobs)} jobs", total=len(jobs))
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(_run_one, *job, capture=True): job for job in jobs
                }
                for fut in as_completed(futures):
                    *_, out_path = futures[fut]
                    ok, output = fut.result()
                    progress.console.print()  # newline before status so it doesn't collide with bar
                    _report(progress.console, out_path, ok, output)
                    progress.advance(task)
                    if ok:
                        n_ok += 1
                    else:
                        n_fail += 1

    console.print(f"\n[bold]{n_ok} completed, {n_fail} failed[/bold]")
