import tqdm
import pandas as pd
from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet
from pathlib import Path

def fix_demo_submission(submission_path, scenario_directory):
    SID_DIGITS = 36
    predictions_df = pd.read_parquet(submission_path)
    fixed_df = pd.DataFrame(columns=predictions_df.columns)

    for id in tqdm.tqdm(predictions_df["scenario_id"].unique()):
        scenario_id = str(id)
        scenario_id = "0"*(SID_DIGITS - len(scenario_id)) + scenario_id
        av2_scenario = load_argoverse_scenario_parquet(scenario_directory / scenario_id / f"scenario_{scenario_id}.parquet")
        
        local_predictions = predictions_df[predictions_df["scenario_id"] == scenario_id]
        for _, pred in local_predictions.iterrows():
        
            # Predictions are all starting at focal agent even if for others
            if pred["probability"] == 1.0:
                for track in av2_scenario.tracks:
                    if track.track_id == pred["track_id"]:
                        xs = pred["predicted_trajectory_x"]
                        ys = pred["predicted_trajectory_y"]

                        fixed_df = pd.concat([fixed_df, pd.DataFrame({
                            "scenario_id": scenario_id,
                            "track_id": pred["track_id"],
                            "probability": pred["probability"],
                            "predicted_trajectory_x": xs - xs[0] + track.object_states[-1].position[0],
                            "predicted_trajectory_y": ys - ys[0] + track.object_states[-1].position[1]
                        })])
                        break

    fixed_df.to_parquet("/path/to/DeMo_submissions_all_translated.parquet")

fix_demo_submission(
    submission_path=Path("/path/to/DeMo_submission_all.parquet"),
    scenario_directory=Path("/path/to/test/raw")
)