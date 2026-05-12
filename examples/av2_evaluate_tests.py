"""
Script to evaluate the submission against the ground truth annotations for the AV2 motion forecasting challenge.
"""

from av2.datasets.motion_forecasting.eval import metrics

import pandas as pd
import numpy as np 

from rich.progress import track
from pathlib import Path
import click
import os

test_anno_path = Path("dataset/av2_mf_focal_test_annotations.parquet")
pred_path = Path("dataset/tracs_predictions/test")

@click.command()
@click.option("--gt-path", default=test_anno_path, help="Path to the test annotations parquet file.")
@click.option("--pred-path", default=pred_path, help="Path to the predictions; either a single parquet, or a directory with one parquet per scenario.")
@click.option("--write-to", default=None, help="Path to write the computed metrics dataframe as a parquet file.")
@click.option("--export-path", default=None, help="Path to export the metrics dataframe as a parquet file.")
@click.option("--mode", default="pipeline", type=click.Choice(["submission", "pipeline"]), help="Evaluate a single submission parquet, or a directory produced by the TraCS pipeline.")
def main(gt_path, pred_path, write_to, export_path, mode):

    try:
        gt_path = Path(gt_path)
        pred_path = Path(pred_path)
    except Exception as e:
        print(f"Error occurred while parsing paths: {e}")
        return

    gt_df = pd.read_parquet(gt_path)
    gt_df_grouped = gt_df.groupby(["scenario_id", "track_id"])

    rows = []

    def directory_iterator(path):
        pred_files = sorted([f for f in os.listdir(path) if f.endswith(".parquet")])
        for file in pred_files:
            pred = pd.read_parquet(path / file)
            scenario_id = file[:-8] # no .parquet ending
            track_id = pred.iloc[0]["track_id"]
            yield (pred, scenario_id, track_id)

    def one_parquet_iterator(path):
        pred_df = pd.read_parquet(path)
        pred_df.rename(columns={"predicted_trajectory_x": "x_traj", "predicted_trajectory_y": "y_traj"}, inplace=True)
        gb = pred_df.groupby(["scenario_id", "track_id"])
        for (scenario_id, track_id), pred in gb:
            yield (pred, scenario_id, track_id)

    match mode:
        case "submission":
            keys = ["probability"]
            iterator = one_parquet_iterator(pred_path)
        case "pipeline":
            keys = ["probability", "tracs_weight_normalized", "tracs_factor"]
            iterator = directory_iterator(pred_path)

    for (pred, scenario_id, track_id) in track(iterator, description="Evaluating submissions", transient=True): 
        if (scenario_id, track_id) not in gt_df_grouped.groups:
            continue # only score focal tracks; focal annotation contains only focal truths
        gt = gt_df_grouped.get_group((scenario_id, track_id))
        
        pred_track = np.moveaxis(
            np.array(
                [[* pred["x_traj"]], 
                [*pred["y_traj"]]]
            ),
            source=[0, 1, 2], 
            destination=[2, 0, 1],
        )

        gt_track = np.array([gt["gt_trajectory_x"].iloc[0], gt["gt_trajectory_y"].iloc[0]]).T
        ade = metrics.compute_ade(pred_track, gt_track)
        fde = metrics.compute_fde(pred_track, gt_track)
        is_missed = metrics.compute_is_missed_prediction(pred_track, gt_track)

        for key in keys:
            pred_prob = pred[key].to_numpy()
            most_prob_index = pred_prob.argmax()

            brier_ade = metrics.compute_brier_ade(pred_track, gt_track, pred_prob)
            brier_fde = metrics.compute_brier_fde(pred_track, gt_track, pred_prob)

            metrics_df_row = {
                "scenario_id": scenario_id,
                "track_id": track_id,
                "key": key,
                "minADE6": ade.min(),
                "minFDE6": fde.min(),
                "minADE1": ade[most_prob_index],
                "minFDE1": fde[most_prob_index],
                "brier_minADE6": brier_ade[ade.argmin()],
                "brier_minFDE6": brier_fde[fde.argmin()],
                "brier_minADE1": brier_ade[most_prob_index],
                "brier_minFDE1": brier_fde[most_prob_index],
                "is_missed_prediction6": is_missed.min(),
                "is_missed_prediction1": is_missed[most_prob_index],
            }
            rows.append(metrics_df_row)

    metrics_df = pd.DataFrame(rows)
    result: pd.DataFrame = metrics_df[["minADE6", "brier_minADE6", "brier_minFDE6", "minFDE6", "is_missed_prediction6", "minADE1", "brier_minADE1", "brier_minFDE1", "minFDE1", "is_missed_prediction1"]].mean()
    print(result)
    if write_to is not None:
        try:
            with open(write_to, "w") as f:
                    print(pred_path.name, gt_path.name, file=f)
                    for key in keys:
                        print(key, file=f)
                        result = metrics_df[metrics_df["key"] == key][["minADE6", "brier_minADE6", "brier_minFDE6", "minFDE6", "is_missed_prediction6", "minADE1", "brier_minADE1", "brier_minFDE1", "minFDE1", "is_missed_prediction1"]]
                        print(result.mean(), file=f)
        except Exception as e:
            print(f"Error occurred while writing to file: {e}")
    if export_path is not None:
        try:
            metrics_df.to_parquet(export_path, index=False)
        except Exception as e:
            print(f"Error occurred while writing to file: {e}")
        print(f"Metrics dataframe exported to {export_path}")


if __name__ == "__main__":
    main()
