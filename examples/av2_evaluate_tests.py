"""
Script to evaluate the submission against the ground truth annotations for the AV2 motion forecasting challenge.
"""

from av2.datasets.motion_forecasting.eval import metrics

import pandas as pd
import numpy as np 

from tqdm import tqdm
from pathlib import Path
import click
import os

test_anno_path = Path("dataset/av2_mf_focal_test_annotations.parquet")
# submission_path = Path("dataset/DeMo_submission_all.parquet")
submission_path = Path("dataset/tracs_predictions/test")

KEYS = ["probability", "tracs_weight_normalized", "tracs_factor"]

@click.command()
@click.option("--gt-path", default=test_anno_path, help="Path to the test annotations parquet file.")
@click.option("--sub-path", default=submission_path, help="Path to the submission parquet file.")
@click.option("--write-to", default=None, help="Path to write the computed metrics dataframe as a parquet file.")
@click.option("--export-path", default=None, help="Path to export the metrics dataframe as a parquet file.")
def main(gt_path, sub_path, write_to, export_path):

    try:
        gt_path = Path(gt_path)
        sub_path = Path(sub_path)
    except Exception as e:
        print(f"Error occurred while parsing paths: {e}")
        return

    gt_df = pd.read_parquet(gt_path)

    gt_df_grouped = gt_df.groupby(["scenario_id", "track_id"])


    metrics_df = pd.DataFrame(columns=["scenario_id", "track_id", "minADE6", "minFDE6", "brier_minADE6", "brier_minFDE6", "is_missed_prediction6", "minADE1", "minFDE1", "brier_minADE1", "brier_minFDE1", "is_missed_prediction1"])

    for file in os.listdir(sub_path):
        pred = pd.read_parquet(sub_path / file)
        scenario_id = file[:-8] # no .parquet ending
        track_id = pred.iloc[0]["track_id"]
        gt = gt_df_grouped.get_group((scenario_id, track_id))
        
        pred_track = np.moveaxis(
            np.array(
                [[* pred["x_traj"]], 
                [*pred["y_traj"]]]
            ),
            source=[0, 1, 2], 
            destination=[2, 0, 1],
        )

        for key in KEYS:
            pred_prob = pred[key].to_numpy()
            most_prob_index = pred_prob.argmax()

            gt_track = np.array([gt["gt_trajectory_x"].iloc[0], gt["gt_trajectory_y"].iloc[0]]).T

            ade = metrics.compute_ade(pred_track, gt_track)
            fde = metrics.compute_fde(pred_track, gt_track)
            
            metrics_df_row = {
                "scenario_id": scenario_id,
                "track_id": track_id,
                "key": key,
                "minADE6": ade.min(),
                "minFDE6": fde.min(),
                "minADE1": ade[most_prob_index],
                "minFDE1": fde[most_prob_index],
                "brier_minADE6": ade[ade.argmin()] + (1.0 - pred_prob[ade.argmin()])**2,
                "brier_minFDE6": fde[fde.argmin()] + (1.0 - pred_prob[fde.argmin()])**2,
                "brier_minADE1": ade[most_prob_index] + (1.0 - pred_prob[most_prob_index])**2,
                "brier_minFDE1": fde[most_prob_index] + (1.0 - pred_prob[most_prob_index])**2,
                "is_missed_prediction6": metrics.compute_is_missed_prediction(pred_track, gt_track).min(),
                "is_missed_prediction1": metrics.compute_is_missed_prediction(pred_track, gt_track)[most_prob_index],
            }
            metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics_df_row, index=[0])], ignore_index=True)

    result: pd.DataFrame = metrics_df[["minADE6", "brier_minADE6", "brier_minFDE6", "minFDE6", "is_missed_prediction6", "minADE1", "brier_minADE1", "brier_minFDE1", "minFDE1", "is_missed_prediction1"]].mean()
    print(result)
    if write_to is not None:
        try:
            with open(write_to, "w") as f:
                    print(sub_path.name, gt_path.name, file=f)
                    for key in KEYS:
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

if __name__ == "__main__":
    main()
