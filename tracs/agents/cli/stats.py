"""CLI script to report runtime and token statistics over a set of TraCS JSON traces."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from rich.console import Console
from rich.table import Table


_GROUP_BY_CHOICES = ["coder_model", "judge_model", "coder_backend", "judge_backend", "scenario", "seed"]

_RUN_METRICS = [
    ("run pipeline seconds", lambda t, its: t["timing"]["pipeline_seconds"]),
    ("run wall seconds", lambda t, its: t["timing"]["total_seconds"]),
    ("run coder seconds", lambda t, its: t["timing"]["coder_seconds"]),
    ("run judge seconds", lambda t, its: t["timing"]["judge_seconds"]),
    ("run total tokens", lambda t, its: t["tokens"]["total_tokens"]),
    ("run coder tokens", lambda t, its: t["tokens"]["coder"]["total_tokens"]),
    ("run judge tokens", lambda t, its: t["tokens"]["judge"]["total_tokens"]),
    ("iterations per run", lambda t, its: len(its)),
]

_ITER_METRICS = [
    ("iteration seconds", lambda it: it["seconds"]),
    ("coder seconds", lambda it: it["coder"]["seconds"]),
    ("judge seconds", lambda it: it["judge"]["seconds"]),
    ("coder output tokens (per answer)", lambda it: it["coder"].get("output_tokens")),
    ("coder input tokens", lambda it: it["coder"].get("input_tokens")),
    ("judge output tokens (per answer)", lambda it: it["judge"].get("output_tokens")),
    ("judge input tokens", lambda it: it["judge"].get("input_tokens")),
]


def _iter_trace_paths(paths: list) -> list:
    files = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        else:
            files.append(p)
    return files


def _load_traces(paths: list) -> tuple[list, int]:
    """Return (traces_with_timing, n_skipped_without_timing)."""
    traces = []
    n_skipped = 0
    for path in _iter_trace_paths(paths):
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"Error: no such file or directory: {path}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError:
            continue
        if "timing" not in trace or "tokens" not in trace:
            n_skipped += 1
            continue
        traces.append(trace)
    return traces, n_skipped


def _summarize(values: list) -> Optional[dict]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    arr = np.asarray(clean, dtype=float)
    q5, q50, q95, q99 = np.percentile(arr, [5, 50, 95, 99])
    return {
        "n": len(clean),
        "mean": arr.mean(),
        "std": arr.std(ddof=1) if len(clean) > 1 else 0.0,
        "q5": q5, "q50": q50, "q95": q95, "q99": q99,
    }


def _collect_metrics(traces: list) -> dict:
    metrics = {name: [] for name, _ in _RUN_METRICS}
    metrics.update({name: [] for name, _ in _ITER_METRICS})
    for trace in traces:
        iterations = trace["iterations"]
        for name, fn in _RUN_METRICS:
            metrics[name].append(fn(trace, iterations))
        for it in iterations:
            for name, fn in _ITER_METRICS:
                metrics[name].append(fn(it))
    return metrics


def _print_table(console: Console, title: str, metrics: dict) -> None:
    table = Table(title=title)
    table.add_column("metric")
    for col in ("n", "mean", "std", "q5", "q50", "q95", "q99"):
        table.add_column(col, justify="right")

    for name, values in metrics.items():
        summary = _summarize(values)
        if summary is None:
            table.add_row(name, "0", "-", "-", "-", "-", "-")
            continue
        table.add_row(
            name,
            str(summary["n"]),
            f"{summary['mean']:.2f}",
            f"{summary['std']:.2f}",
            f"{summary['q5']:.2f}",
            f"{summary['q50']:.2f}",
            f"{summary['q95']:.2f}",
            f"{summary['q99']:.2f}",
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tracs_agents_stats",
        description="Report runtime and token statistics over a set of TraCS JSON trace files.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Trace .json file(s) and/or directories containing trace .json files.",
    )
    parser.add_argument(
        "--group-by",
        default="coder_model",
        choices=_GROUP_BY_CHOICES,
        metavar="KEY",
        help=f"Parameter to break the per-group tables down by (default: coder_model). Choices: {', '.join(_GROUP_BY_CHOICES)}.",
    )
    args = parser.parse_args()

    console = Console(highlight=False)

    traces, n_skipped = _load_traces(args.paths)
    if not traces:
        console.print("[red]Error:[/red] no traces with timing data found.")
        sys.exit(1)

    n_iterations = sum(len(t["iterations"]) for t in traces)
    console.print(
        f"Traces: {len(traces)} ({n_skipped} without timing data, skipped) · {n_iterations} iterations\n"
    )

    _print_table(console, "Overall", _collect_metrics(traces))

    groups: dict = {}
    for trace in traces:
        key = trace["parameters"].get(args.group_by)
        groups.setdefault(key, []).append(trace)

    for key in sorted(groups, key=lambda k: (k is None, k)):
        console.print()
        _print_table(console, f"{args.group_by} = {key}", _collect_metrics(groups[key]))


if __name__ == "__main__":
    main()
