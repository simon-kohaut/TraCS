"""AV2 motion-forecasting metrics and TraCS scoring utilities."""

#
# Copyright (c) Simon Kohaut, Felix Divo, Julius Hahnewald, and contributors
#
# This file is part of TraCS and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with TraCS.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Standard Library
from pathlib import Path

# Third Party
import numpy as np
import pandas as pd
from av2.datasets.motion_forecasting.eval import metrics

# TraCS
from tracs.av2.av2_loader import load_focal_gt_trajectory


def tracs_pointwise_to_score(pointwise_scores: np.ndarray) -> float:
    N = len(pointwise_scores)
    return np.exp(np.sum(np.log(np.clip(pointwise_scores, 1e-8, 1.0))) / N)


def compute_argoverse2_metrics(
    scored_predictions: pd.DataFrame,
    data_root: Path,
    split: str,
    scenario_id: str,
) -> pd.DataFrame:
    # Backbone, TraCS alone, Backbone & TraCS
    methods = ["backbone_trajectory_score", "tracs_trajectory_score", "tracs_backbone_independent_op", "tracs_backbone_linear_op"]

    # Get ground truth trajectory for comparison
    gt_x, gt_y = load_focal_gt_trajectory(data_root, split, scenario_id)
    gt_track = np.array([gt_x, gt_y]).T

    predicted_tracks = np.moveaxis(
        np.array(
            [[*scored_predictions["x_traj"]],
            [*scored_predictions["y_traj"]]]
        ),
        source=[0, 1, 2],
        destination=[2, 0, 1],
    )

    rows = []

    ade = metrics.compute_ade(predicted_tracks, gt_track)
    fde = metrics.compute_fde(predicted_tracks, gt_track)
    is_missed = metrics.compute_is_missed_prediction(predicted_tracks, gt_track)

    for method in methods:
        weights = scored_predictions[method].to_numpy()
        most_prob_index = weights.argmax()

        brier_ade = metrics.compute_brier_ade(predicted_tracks, gt_track, weights)
        brier_fde = metrics.compute_brier_fde(predicted_tracks, gt_track, weights)

        metrics_df_row = {
            "scenario_id": scenario_id,
            "track_id": scored_predictions.iloc[0]["track_id"],
            "method": method,
            "minADE6": ade.min(),
            "minFDE6": fde.min(),
            "minADE1": ade[most_prob_index],
            "minFDE1": fde[most_prob_index],
            "brier_minADE6": brier_ade[ade.argmin()],
            "brier_minFDE6": brier_fde[fde.argmin()],
            "brier_minADE1": brier_ade[most_prob_index],
            "brier_minFDE1": brier_fde[most_prob_index],
            "hit_rate": 1 - is_missed.mean(),
            "is_missed_prediction6": is_missed.min(),
            "is_missed_prediction1": is_missed[most_prob_index],
        }
        rows.append(metrics_df_row)

    return pd.DataFrame(rows)
