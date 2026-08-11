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
    sampled_pct = 100.0 * k / len(all_ids)

    console = Console()
    console.print(
        f"Sampling {k} of {len(all_ids)} scenarios ({sampled_pct:.2f}%) uniformly at random "
        f"from the '{args.split}' split."
    )
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
            try:
                scored_predictions, _ = tracs_rate_scenario(
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
                # Recorded with no timing so the failure survives into the output CSV
                records.append(
                    {
                        "scenario_id": scenario_id,
                        "seconds": np.nan,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                progress.advance(task)
                continue
            # Deployment-relevant inference time only: excludes scenario loading and grading,
            # and nets out the sleep barrier. See tracs_rate_scenario's `inference_seconds` attr.
            inference_seconds = scored_predictions.attrs["inference_seconds"]
            records.append({"scenario_id": scenario_id, "seconds": inference_seconds, "error": None})
            progress.advance(task)

    times = np.array([r["seconds"] for r in records if r["error"] is None])
    n_failed = len(records) - len(times)
    if n_failed:
        console.print(f"[warn] {n_failed} of {len(records)} scenarios failed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    if len(times) == 0:
        # Still write the file: the per-scenario error messages are the whole result
        df.to_csv(args.output, index=False)
        console.print("[red]No scenarios completed successfully; nothing to report.")
        console.print(f"Wrote per-scenario failures to {args.output}")
        return

    mean = np.mean(times)
    std = np.std(times, ddof=1) if len(times) > 1 else 0.0
    # Delta method: Hz = 1/t is nonlinear, so std(Hz) is not 1/std(t); propagate instead via
    # std(f(t)) ~= |f'(mean(t))| * std(t) = std(t) / mean(t)**2.
    std_hz = std / mean**2
    q5, q50, q95, q99 = np.percentile(times, [5, 50, 95, 99])

    table = Table(
        title=(
            f"TraCS inference runtime per scenario "
            f"({len(times)} of {len(all_ids)} scenarios in '{args.split}', {sampled_pct:.2f}% sampled)"
        )
    )
    table.add_column("Statistic")
    table.add_column("Value (s)", justify="right")
    table.add_column("Value (Hz)", justify="right")
    table.add_row("mean", f"{mean:.4f}", f"{1 / mean:.4f}")
    table.add_row("std", f"{std:.4f}", f"{std_hz:.4f}")
    table.add_row("q5", f"{q5:.4f}", f"{1 / q5:.4f}")
    table.add_row("q50 (median)", f"{q50:.4f}", f"{1 / q50:.4f}")
    table.add_row("q95", f"{q95:.4f}", f"{1 / q95:.4f}")
    table.add_row("q99", f"{q99:.4f}", f"{1 / q99:.4f}")
    console.print(table)

    summary = {
        "n_scenarios": len(times),
        "n_failed": n_failed,
        "n_total_in_split": len(all_ids),
        "sampled_pct": sampled_pct,
        "mean": mean,
        "mean_hz": 1 / mean,
        "std": std,
        "std_hz": std_hz,
        "q5": q5,
        "q5_hz": 1 / q5,
        "q50": q50,
        "q50_hz": 1 / q50,
        "q95": q95,
        "q95_hz": 1 / q95,
        "q99": q99,
        "q99_hz": 1 / q99,
    }
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
