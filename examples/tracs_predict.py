"""Batch-evaluate TraCS logic programs over an AV2 split with crash-resume support."""

import multiprocessing
import os
import random
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rich.progress import MofNCompleteColumn, Progress, ProgressColumn
from rich.text import Text

from eval import tracs_rate_scenario


class IterationsPerSecondColumn(ProgressColumn):
    def render(self, task) -> Text:
        speed = task.speed
        if speed is None:
            return Text("? it/s", style="progress.data.speed")
        return Text(f"{speed:.2f} it/s", style="progress.data.speed")


def process_scenario(
    scenario_id: str,
    logic: str,
    prediction_path: Path,
    data_root: Path,
    split: str,
):
    scored_predictions, _ = tracs_rate_scenario(
        scenario_id,
        logic=logic,
        prediction_path=prediction_path,
        data_root=data_root,
        split=split,
    )
    scored_predictions.insert(0, "scenario_id", scenario_id)
    return scored_predictions


class ResumableParquetSink:
    """Buffers scenario DataFrames in memory and periodically rewrites the destination
    parquet file (existing rows + buffer) atomically. The on-disk file is always a valid,
    sealed parquet; a SIGKILL only loses the unflushed buffer."""

    def __init__(self, final_path: Path, flush_every: int):
        self.final_path = final_path
        self.tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        self.flush_every = flush_every
        self._buffer: list = []
        self._file_present = final_path.exists()

    def done_ids(self) -> set[str]:
        if not self._file_present:
            return set()
        with pq.ParquetFile(self.final_path) as pf:
            col = pf.read(columns=["scenario_id"])["scenario_id"]
        return set(col.to_pylist())

    def write(self, df) -> None:
        self._buffer.append(pa.Table.from_pandas(df, preserve_index=False))
        if len(self._buffer) >= self.flush_every:
            self._flush()

    def close(self) -> None:
        if self._buffer:
            self._flush()

    def _flush(self) -> None:
        tables = self._buffer
        if self._file_present:
            with pq.ParquetFile(self.final_path) as pf:
                tables = [pf.read()] + tables
        pq.write_table(pa.concat_tables(tables), self.tmp_path, row_group_size=10_000)
        os.replace(self.tmp_path, self.final_path)
        self._file_present = True
        self._buffer.clear()


def main(args) -> None:
    submission_file = (
        args.data_root
        / "av2_backbone_outputs"
        / args.backbone
        / f"submission_{args.split}.parquet"
    )
    final_path: Path = args.output_dir / args.logic.stem / args.backbone / f"{args.split}.parquet"
    final_path.parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        logic=args.logic.read_text(),
        prediction_path=submission_file,
        data_root=args.data_root,
        split=args.split,
    )

    sink = ResumableParquetSink(final_path, flush_every=args.flush_every)
    all_ids = sorted(
        file.stem for file in (args.data_root / "av2" / args.split).iterdir()
    )
    if args.subsample < 1.0:
        k = max(1, round(len(all_ids) * args.subsample))
        all_ids = random.Random(123).sample(all_ids, k)
    done = sink.done_ids()
    scenario_ids = [sid for sid in all_ids if sid not in done]

    def results_iter():
        if args.workers is not None and args.workers > 1:
            with multiprocessing.Pool(args.workers) as p:
                yield from p.imap_unordered(
                    partial(process_scenario, **kwargs), scenario_ids
                )
        else:
            for sid in scenario_ids:
                yield process_scenario(sid, **kwargs)

    try:
        with Progress(
            *Progress.get_default_columns(),
            MofNCompleteColumn(),
            IterationsPerSecondColumn(),
        ) as progress:
            task = progress.add_task(
                "Processing scenarios...", total=len(all_ids), completed=len(done)
            )
            for df in results_iter():
                sink.write(df)
                progress.advance(task)
    except KeyboardInterrupt:
        print("Process interrupted by user. Flushing and exiting...")
    finally:
        sink.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--split", type=str, default="val", help="AV2 split to evaluate (train/val/test).")
    parser.add_argument("--backbone", type=str, default="DeMo", help="Backbone name; used to locate the submission parquet and as a path component in the output.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data") / "tracs_predictions",
        help="Parent directory for outputs; the actual file is <output_dir>/<logic>/<backbone>/<split>.parquet.",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=500,
        help="Re-seal the parquet file every N scenarios for crash-resume safety.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Fraction of scenarios to evaluate (0 < subsample <= 1). Uses a fixed seed for reproducibility.",
    )
    parser.add_argument(
        "--logic",
        type=Path,
        default=Path("logic_programs/simplistic.resin"),
        help="Path to the Resin logic program to evaluate. The stem is appended to --output_dir, so different logic files land in separate subdirectories automatically.",
    )
    args = parser.parse_args()

    main(args)
