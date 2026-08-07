"""Benchmark per-scenario TraCS inference runtime over a deterministic subsample of a split."""

#
# Copyright (c) Simon Kohaut, Felix Divo, Julius Hahnewald, and contributors
#
# This file is part of TraCS and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with TraCS.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from tracs.av2.inference import tracs_rate_scenario


class _IterationsPerSecondColumn(ProgressColumn):
    def render(self, task) -> Text:
        speed = task.speed
        if speed is None:
            return Text("? it/s", style="progress.data.speed")
        return Text(f"{speed:.2f} it/s", style="progress.data.speed")


def _run(args: argparse.Namespace) -> None:
    submission_file = (
        args.data_root
        / "av2_backbone_outputs"
        / args.backbone
        / f"submission_{args.split}.parquet"
    )
    logic = args.logic.read_text()

    all_ids = sorted(
        file.stem for file in (args.data_root / "av2" / args.split).iterdir()
    )
    k = max(1, round(len(all_ids) * args.subsample))
    scenario_ids = random.Random(args.seed).sample(all_ids, k)

    console = Console()
    records: list[dict] = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
        _IterationsPerSecondColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Benchmarking scenarios...", total=len(scenario_ids))
        for scenario_id in scenario_ids:
            start = time.perf_counter()
            try:
                tracs_rate_scenario(
                    scenario_id,
                    logic=logic,
                    prediction_path=submission_file,
                    data_root=args.data_root,
                    split=args.split,
                )
            except Exception as exc:
                console.print(
                    f"[warn] {scenario_id} failed ({type(exc).__name__}: {exc}), skipping"
                )
                progress.advance(task)
                continue
            elapsed = time.perf_counter() - start
            records.append({"scenario_id": scenario_id, "seconds": elapsed})
            progress.advance(task)

    times = np.array([r["seconds"] for r in records])
    if len(times) == 0:
        console.print("[red]No scenarios completed successfully; nothing to report.")
        return

    mean = np.mean(times)
    std = np.std(times, ddof=1) if len(times) > 1 else 0.0
    q5, q50, q95, q99 = np.percentile(times, [5, 50, 95, 99])

    table = Table(title=f"TraCS inference runtime per scenario ({len(times)} scenarios)")
    table.add_column("Statistic")
    table.add_column("Value (s)", justify="right")
    table.add_row("mean", f"{mean:.4f}")
    table.add_row("std", f"{std:.4f}")
    table.add_row("q5", f"{q5:.4f}")
    table.add_row("q50 (median)", f"{q50:.4f}")
    table.add_row("q95", f"{q95:.4f}")
    table.add_row("q99", f"{q99:.4f}")
    table.add_row("total wall time", f"{times.sum():.4f}")
    console.print(table)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_scenarios": len(times),
        "mean": mean,
        "std": std,
        "q5": q5,
        "q50": q50,
        "q95": q95,
        "q99": q99,
    }
    df = pd.DataFrame(records)
    for key, value in summary.items():
        df[key] = value
    df.to_csv(args.output, index=False)
    console.print(f"Wrote per-scenario timings and summary to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tracs_benchmark_inference",
        description="Benchmark per-scenario TraCS inference runtime over a deterministic subsample of an AV2 split.",
    )
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--split", type=str, default="test", help="AV2 split to benchmark (train/val/test).")
    parser.add_argument("--backbone", type=str, default="DeMo", help="Backbone name; used to locate the submission parquet.")
    parser.add_argument(
        "--logic",
        type=Path,
        default=Path("logic_programs/complete2.resin"),
        help="Path to the Resin logic program to evaluate.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.1,
        help="Fraction of scenarios in the split to benchmark (0 < subsample <= 1). Deterministic given --seed.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed for deterministic scenario subsampling.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/benchmark_inference.csv"),
        help="CSV file to write per-scenario timings and summary statistics to.",
    )
    args = parser.parse_args()

    _run(args)


if __name__ == "__main__":
    main()
