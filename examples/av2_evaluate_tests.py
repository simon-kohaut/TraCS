"""Evaluate AV2 motion-forecasting submissions against ground-truth annotations."""

from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.progress import track

from tracs.av2.metrics import compute_weighted_metrics, metrics


METRIC_COLS = [
    "minADE6", "brier_minADE6", "brier_minFDE6", "minFDE6", "is_missed_prediction6",
    "minADE1", "brier_minADE1", "brier_minFDE1", "minFDE1", "is_missed_prediction1",
]


def _iter_directory(path: Path):
    for parquet in sorted(path.glob("*.parquet")):
        pred = pd.read_parquet(parquet)
        yield pred, parquet.stem, pred.iloc[0]["track_id"]


def _iter_submission(path: Path):
    pred_df = pd.read_parquet(path).rename(columns={
        "predicted_trajectory_x": "x_traj",
        "predicted_trajectory_y": "y_traj",
    })
    for (scenario_id, track_id), pred in pred_df.groupby(["scenario_id", "track_id"]):
        yield pred, scenario_id, track_id


@click.command()
@click.option("--gt-path", default="data/av2/av2_mf_focal_test_annotations.parquet", type=click.Path(exists=True, path_type=Path), help="Path to the test annotations parquet file.")
@click.option("--pred-path", default="data/tracs_predictions/test", type=click.Path(exists=True, path_type=Path), help="Path to the predictions; either a single parquet, or a directory with one parquet per scenario.")
@click.option("--write-to", default=None, type=click.Path(path_type=Path), help="Path to write a per-key summary as a text file.")
@click.option("--export-path", default=None, type=click.Path(path_type=Path), help="Path to export the full metrics dataframe as a parquet file.")
@click.option("--mode", default="pipeline", type=click.Choice(["submission", "pipeline"]), help="Evaluate a single submission parquet, or a directory produced by the TraCS pipeline.")
def main(gt_path: Path, pred_path: Path, write_to: Path | None, export_path: Path | None, mode: str) -> None:
    gt_df_grouped = pd.read_parquet(gt_path).groupby(["scenario_id", "track_id"])

    match mode:
        case "submission":
            keys, iterator = ["probability"], _iter_submission(pred_path)
        case "pipeline":
            keys, iterator = ["probability", "tracs_weight_normalized", "tracs_factor"], _iter_directory(pred_path)

    rows = []
    for pred, scenario_id, track_id in track(iterator, description="Evaluating submissions", transient=True):
        if (scenario_id, track_id) not in gt_df_grouped.groups:
            continue  # only score focal tracks

        gt = gt_df_grouped.get_group((scenario_id, track_id))
        pred_track = np.moveaxis(
            np.array([[*pred["x_traj"]], [*pred["y_traj"]]]),
            source=[0, 1, 2], destination=[2, 0, 1],
        )
        gt_track = np.array([gt["gt_trajectory_x"].iloc[0], gt["gt_trajectory_y"].iloc[0]]).T

        ade = metrics.compute_ade(pred_track, gt_track)
        fde = metrics.compute_fde(pred_track, gt_track)
        is_missed = metrics.compute_is_missed_prediction(pred_track, gt_track)

        for key in keys:
            row = compute_weighted_metrics(pred_track, gt_track, pred[key].to_numpy(), ade, fde, is_missed)
            row |= {"scenario_id": scenario_id, "track_id": track_id, "key": key}
            rows.append(row)

    metrics_df = pd.DataFrame(rows)
    print(metrics_df[METRIC_COLS].mean())

    if write_to is not None:
        with open(write_to, "w") as f:
            print(pred_path.name, gt_path.name, file=f)
            for key in keys:
                print(key, file=f)
                print(metrics_df[metrics_df["key"] == key][METRIC_COLS].mean(), file=f)

    if export_path is not None:
        metrics_df.to_parquet(export_path, index=False)
        print(f"Metrics dataframe exported to {export_path}")


if __name__ == "__main__":
    main()
