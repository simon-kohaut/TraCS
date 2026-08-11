"""Benchmark FusionNet forward-pass latency (batch size 1, CPU and GPU) over fixed-size feature vectors."""

#
# Copyright (c) Simon Kohaut, Felix Divo, Julius Hahnewald, and contributors
#
# This file is part of TraCS and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with TraCS.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

from tracs.fusion import FusionNet

_N_WARMUP_STEPS = 1000


def _time_forward_passes(model: FusionNet, x: torch.Tensor, n_steps: int, is_cuda: bool) -> np.ndarray:
    timings = np.empty(n_steps)
    for step in range(n_steps):
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(x)
        if is_cuda:
            torch.cuda.synchronize()
        timings[step] = time.perf_counter() - start

    return timings


def _report(console: Console, device_name: str, times: np.ndarray) -> dict:
    mean = np.mean(times)
    std = np.std(times, ddof=1) if len(times) > 1 else 0.0
    # Delta method: Hz = 1/t is nonlinear, so std(Hz) is not 1/std(t); propagate instead via
    # std(f(t)) ~= |f'(mean(t))| * std(t) = std(t) / mean(t)**2.
    std_hz = std / mean**2
    q5, q50, q95, q99 = np.percentile(times, [5, 50, 95, 99])

    table = Table(title=f"FusionNet forward-pass latency, batch size 1 ({device_name}, n={len(times)})")
    table.add_column("Statistic")
    table.add_column("Value (s)", justify="right")
    table.add_column("Value (Hz)", justify="right")
    table.add_row("mean", f"{mean:.6f}", f"{1 / mean:.2f}")
    table.add_row("std", f"{std:.6f}", f"{std_hz:.2f}")
    table.add_row("q5", f"{q5:.6f}", f"{1 / q5:.2f}")
    table.add_row("q50 (median)", f"{q50:.6f}", f"{1 / q50:.2f}")
    table.add_row("q95", f"{q95:.6f}", f"{1 / q95:.2f}")
    table.add_row("q99", f"{q99:.6f}", f"{1 / q99:.2f}")
    console.print(table)

    return {
        "mean": mean, "mean_hz": 1 / mean, "std": std, "std_hz": std_hz,
        "q5": q5, "q5_hz": 1 / q5,
        "q50": q50, "q50_hz": 1 / q50,
        "q95": q95, "q95_hz": 1 / q95,
        "q99": q99, "q99_hz": 1 / q99,
    }


def _run(args: argparse.Namespace) -> None:
    console = Console()

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    else:
        console.print("[warn] CUDA not available; skipping GPU variant.")

    # Forward-pass latency depends only on tensor shape, not on the values flowing through it, so a
    # single fixed-size random vector -- reused for both warmup and measurement -- times the model
    # identically to real per-scenario features while skipping the (expensive, ProMis-driven) work of
    # producing those features.
    n_emb = max(1, args.feature_pool_size // 2)

    all_records = []
    for device in devices:
        device_name = device.type
        console.print(f"\nBenchmarking on {device_name}...")

        torch.manual_seed(args.seed)
        model = FusionNet(args.feature_pool_size, n_emb=n_emb).to(device)
        model.eval()

        x = torch.randn(1, args.feature_pool_size, device=device)
        is_cuda = device.type == "cuda"
        _time_forward_passes(model, x, _N_WARMUP_STEPS, is_cuda)  # warmup, untimed
        times = _time_forward_passes(model, x, args.n_measurement_steps, is_cuda)

        summary = _report(console, device_name, times)
        for step, seconds in enumerate(times):
            all_records.append({"device": device_name, "step": step, "seconds": seconds, **summary})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_records).to_csv(args.output, index=False)
    console.print(f"\nWrote per-step timings and summary to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tracs_benchmark_fusion",
        description=(
            "Benchmark FusionNet forward-pass latency (batch size 1, CPU and GPU) over a fixed "
            "1000-step warmup and a configurable measurement phase, using a fixed-size feature vector."
        ),
    )
    parser.add_argument(
        "--feature_pool_size",
        type=int,
        default=50,
        help="Dimensionality of the (synthetic) FusionNet input feature vector.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed for the model's random init and input vector.")
    parser.add_argument(
        "--n_measurement_steps",
        type=int,
        default=1000,
        help="Number of timed forward passes to run (after a fixed 1000-step warmup).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/benchmark_fusion.csv"),
        help="CSV file to write per-step timings and summary statistics to.",
    )
    args = parser.parse_args()

    _run(args)


if __name__ == "__main__":
    main()
