# Standard Library
import copy
from pathlib import Path
import time

# Third Party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import av2.rendering.vector as vector_plotting_utils
from av2.datasets.motion_forecasting.eval import metrics
from av2.map.map_api import ArgoverseStaticMap
from promis import StaRMap, ProMis
from promis.geo import CartesianRasterBand, CartesianMap, CartesianCollection

# TraCS
from tracs.av2_loader import AV2StaticMapLoader, AV2AgentLoader

FEATURE_DESCRIPTION = {
    "crosswalk": "PEDESTRIAN_CROSSING",
    "drivable_area": "DRIVABLE_AREA",
    "intersection": "INTERSECTION",
    "vehicle_lane": "VEHICLE",
    "bike_lane": "BIKE",
    "bus_lane": "BUS",
    "solid_line": ["DOUBLE_SOLID_YELLOW", "DOUBLE_SOLID_WHITE", "SOLID_YELLOW", "SOLID_WHITE",],
    "dashed_line": ["DASHED_WHITE", "DASHED_YELLOW", "DOUBLE_DASH_YELLOW", "DOUBLE_DASH_WHITE",],
} 


def plot_scenario_roadlayout(av2_map, ax=None):
    if ax is None:
        ax = plt.gca()

    # Plot lane segments
    for _, ls in av2_map.vector_lane_segments.items():
        vector_plotting_utils.draw_polygon_mpl(
            ax, ls.polygon_boundary, color="gray", linewidth=0.5
        )
        vector_plotting_utils.plot_polygon_patch_mpl(
            ls.polygon_boundary, ax, color="gray", alpha=0.2
        )

    # Plot pedestrian crossings
    for _, pc in av2_map.vector_pedestrian_crossings.items():
        vector_plotting_utils.draw_polygon_mpl(ax, pc.polygon, color="dimgray", linewidth=0.5)
        vector_plotting_utils.plot_polygon_patch_mpl(
            pc.polygon, ax, color="dimgray", alpha=0.2
        )


def plot_predicted_others(agent_loader, ax=None):
    if ax is None:
        ax = plt.gca()

    agent_loader.reset()
    for timestep in range(60):
        alpha = (timestep + 1) / 60.0
        current_state = agent_loader.features
        agent_loader.update(1)
        next_state = agent_loader.features

        for before, after in zip(current_state, next_state):
            ax.arrow(before.x, before.y, after.x - before.x, after.y - before.y,
                     shape='full', lw=0, length_includes_head=True, color="green",
                     head_width=3.5, head_length=1.5, alpha=alpha)
    agent_loader.reset()


def tracs_pointwise_to_score(pointwise_scores):
    N = len(pointwise_scores)
    return np.exp(np.sum(np.log(np.clip(pointwise_scores, 1e-8, 1.0))) / N)


def tracs_rate_all(
        promis, 
        trajectories,
        speed_writer,
        agent_loader, 
        static_map,
        scenario_length = 60,
    ) -> None:
    agent_loader.reset()
    agent_loader.apply_covariance({"all": np.power(1.639, 2) * np.eye(2)}) #minADE1 DeMo

    num_timesteps = len(trajectories[0]["x_traj"])
    tracs_scores = np.zeros((len(trajectories), num_timesteps))
    for timestep in range(0, num_timesteps):
        # Get next predicted locations and transitions
        agent_loader.update(1)
        promis.star_map.uam.features = static_map.features + agent_loader.features
        agent_locations = CartesianCollection(None)
        positions = np.vstack([np.array([p["x_traj"], p["y_traj"]])[:, timestep] for p in trajectories])

        # At the end of predicted trajectory we repeat transition velocity
        if timestep < scenario_length - 1:
            next_positions = np.vstack([np.array([p["x_traj"], p["y_traj"]])[:, timestep+1] for p in trajectories])
            transitions=np.hstack((next_positions - positions, np.zeros((len(trajectories), 1))))
        
        # Update at trajectory location and transition
        agent_locations.append_with_default(
            coordinates=positions,
            value=0.0,
            transitions=transitions,
        )

        # Write new data via StaR Map
        relations = promis.star_map.get_all()
        for relation in relations:
            for location_type in relations[relation]:
                    promis.star_map.update(relation, location_type, agent_locations, 10)

        if speed_writer is not None:
            distance_traveled_km = np.linalg.norm(transitions / 1000.0, axis=1)
            time_in_hrs = 0.1 / 60 / 60
            speed = distance_traveled_km / time_in_hrs
            speed_writer.write(speed, time.monotonic())

        # Get new inference results and append as tracs scores
        time.sleep(0.01)  # Ensures that all data has already transferred
        promis._evaluation_points = agent_locations
        updated_landscape = promis.update()
        if updated_landscape is not None:
            landscape = updated_landscape

        for i in range(len(trajectories)):
            tracs_scores[i, timestep] = landscape.values()[i, 0]

    return tracs_scores


def tracs_rate_traj(promis, trajectory, speed_writer, agent_loader, static_map, scenario_length: int = 60, last_score = 0):
    # Get trajectory xs and ys
    x_traj = trajectory["x_traj"]
    y_traj = trajectory["y_traj"]

    num_timesteps = len(x_traj)
    agent_locations = CartesianCollection(None)
    agent_locations.append_with_default(np.zeros((1, 2)), 0.0)
    tracs_scores = []
    agent_loader.reset()
    for timestep in range(0, num_timesteps):
        # Update time
        agent_loader.update(1)

        # Get next predicted locations and transitions
        position = np.array([[x_traj[timestep], y_traj[timestep]]])

        # At the end of predicted trajectory we repeat transition velocity
        if timestep < scenario_length - 1:
            next_position = np.array([x_traj[timestep + 1], y_traj[timestep + 1]])
            transition = np.hstack([next_position - position, np.zeros((1, 1))])
        
        # Update at trajectory location and transition
        agent_locations.data.loc[0, "east"] = position[0, 0]
        agent_locations.data.loc[0, "north"] = position[0, 1]
        agent_locations.data.loc[0, "delta_east"] = transition[0, 0]
        agent_locations.data.loc[0, "delta_north"] = transition[0, 1]

        # Write new data via StaR Map
        promis.star_map.uam.features = static_map.features + agent_loader.features
        relations = promis.star_map.get_all()
        for relation in relations:
            for location_type in relations[relation]:
                    promis.star_map.update(relation, location_type, agent_locations, 10, "nearest")

        if speed_writer is not None:
            distance_traveled_km = np.linalg.norm(transition[0, :2] / 1000.0)
            time_in_hrs = 0.1 / 60 / 60
            speed = distance_traveled_km / time_in_hrs
            speed_writer.write(np.array([speed]), time.monotonic())

        # Get new inference results and append as tracs scores
        time.sleep(0.01)  # Ensures that on slow systems or large problems all data has already transferred
        updated_landscape = promis.update()
        if updated_landscape is not None:
            score = updated_landscape.values().item()
            last_score = score
        else:
            score = last_score
        
        tracs_scores.append(score)

    return np.array(tracs_scores)


def tracs_rate_scenario(
        scenario_id: str,
        logic: str, 
        prediction_path: Path,
        data_root: Path,
        mode: str,
        scenario_length: int = 60,
        trust: float = 1.0,
        write_speed = True
    ) -> None:
    static_map = AV2StaticMapLoader(
        path=data_root / mode / "raw" / scenario_id / f"log_map_archive_{scenario_id}.json",
        feature_description=FEATURE_DESCRIPTION,
    ).to_cartesian_map()

    agent_loader = AV2AgentLoader(
        prediction_path=prediction_path,
        split_path=data_root / mode / "raw",
        scenario_id=scenario_id,
    )

    agent_loader.reset()
    agent_loader.apply_covariance({"all": np.power(1.639, 2) * np.eye(2)}) #minADE1 DeMo

    # Initialize StaR Map and ProMis
    raster = CartesianRasterBand(
        origin=None,
        resolution=(7, 1),
        width=1,
        height=1,
    )

    star_map = StaRMap(CartesianMap(None))
    star_map.initialize(
        evaluation_points=raster,
        number_of_random_maps=25,
        logic=logic,
    )

    promis = ProMis(star_map, logic, 7)
    promis.initialize(evaluation_points=raster, interpolation_method="nearest")
    if write_speed:
        speed_writer = promis._resin.make_writer_for("speed")
    else:
        speed_writer = None

    # Get ground truth trajectory and append to predictions for rating
    ground_truth = pd.read_parquet(data_root / "av2_mf_focal_test_annotations.parquet")
    ground_truth_traj = ground_truth[ground_truth["scenario_id"] == scenario_id].iloc[0]
    ground_truth_traj["x_traj"] = ground_truth_traj["gt_trajectory_x"]
    ground_truth_traj["y_traj"] = ground_truth_traj["gt_trajectory_y"]
    trajectories = agent_loader.focal_predictions + [ground_truth_traj]

    scores = tracs_rate_all(promis, trajectories, speed_writer, agent_loader, static_map, scenario_length)
    for i, prediction in enumerate(agent_loader.focal_predictions):
        prediction["tracs_scores"] = scores[i]
        prediction["tracs_score"] = tracs_pointwise_to_score(scores[i])
        prediction["tracs_std"] = scores[i].std()
        prediction["tracs_weight"] = trust * prediction["tracs_score"] + (1 - trust) / 6

    scored_predictions = pd.DataFrame(agent_loader.focal_predictions)
    scored_predictions["tracs_weight_normalized"] = scored_predictions["tracs_weight"] / np.sum(scored_predictions["tracs_weight"])
    scored_predictions["tracs_factor"] = scored_predictions["probability"] * scored_predictions["tracs_weight"] \
        / np.sum(scored_predictions["probability"] * scored_predictions["tracs_weight"] ) # or use only / np.sum() to normalize?

    return scored_predictions, tracs_pointwise_to_score(scores[6])


def show_comparison_study(scored_predictions, gt_score, data_root, mode, scenario_id):
    # Load also other agents (predicted) paths
    agent_loader = AV2AgentLoader(
        prediction_path=data_root / "DeMo_submission_translated_merged.parquet",
        split_path=data_root / mode / "raw",
        scenario_id=scenario_id,
    )

    # Render scenario once per trajectory and scoring method
    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(24, 12), layout='constrained', sharex=True, sharey=True)

    # Load scneario
    av2_map = ArgoverseStaticMap.from_json(data_root / mode / "raw" / scenario_id / f"log_map_archive_{scenario_id}.json")
    ground_truth = pd.read_parquet(data_root / "av2_mf_focal_test_annotations.parquet")
    grount_truth_traj = ground_truth[ground_truth["scenario_id"] == scenario_id].iloc[0]
    
    # Get normalized (scores for better viewing)
    tracs_scores_vals = np.concatenate(scored_predictions["tracs_scores"].values)
    tracs_scores_norm = plt.Normalize(vmin=tracs_scores_vals.min(), vmax=tracs_scores_vals.max())
    backbone_norm = plt.Normalize(
        vmin=scored_predictions["probability"].min(),
        vmax=scored_predictions["probability"].max()
    )
    tracs_norm = plt.Normalize(
        vmin=scored_predictions["tracs_weight_normalized"].min(),
        vmax=scored_predictions["tracs_weight_normalized"].max()
    )
    tracs_backbone_norm = plt.Normalize(
        vmin=scored_predictions["tracs_factor"].min(),
        vmax=scored_predictions["tracs_factor"].max()
    )

    # The actual plotting
    for traj in range(len(scored_predictions)):
        x_traj = scored_predictions.iloc[traj]["x_traj"]
        y_traj = scored_predictions.iloc[traj]["y_traj"]
        n_pts = len(x_traj)

        xmin, xmax = x_traj.min() - 20, x_traj.max() + 20
        ymin, ymax = y_traj.min() - 20, y_traj.max() + 20
        for i in range(4):
            plot_scenario_roadlayout(av2_map, ax=axes[i, traj])
            plot_predicted_others(agent_loader, axes[i, traj])
            axes[i, traj].set_aspect(1)
            axes[i, traj].set_xlim(xmin, xmax)        
            axes[i, traj].set_ylim(ymin, ymax)   

        sc0 = axes[0, traj].scatter(
            x_traj,
            y_traj,
            c=scored_predictions.iloc[traj]["tracs_scores"],
            s=15.0,
            cmap="cool_r", norm=tracs_scores_norm
        )
        sc1 = axes[1, traj].scatter(
            x_traj,
            y_traj,
            c=[scored_predictions.iloc[traj]["tracs_weight_normalized"]] * n_pts,
            s=15.0,
            cmap="cool_r", norm=tracs_norm
        )
        sc2 = axes[2, traj].scatter(
            x_traj,
            y_traj,
            c=[scored_predictions.iloc[traj]["probability"]] * n_pts,
            s=15.0,
            cmap="cool_r", norm=backbone_norm
        )
        sc3 = axes[3, traj].scatter(
            x_traj,
            y_traj,
            c=[scored_predictions.iloc[traj]["tracs_factor"]] * n_pts,
            s=15.0,
            cmap="cool_r", norm=tracs_backbone_norm
        )

        for i in range(4):
            axes[i, traj].plot(grount_truth_traj["gt_trajectory_x"], grount_truth_traj["gt_trajectory_y"], c="black")

    fig.colorbar(sc0, ax=axes[0, :], label="Value")
    fig.colorbar(sc1, ax=axes[1, :], label="Value")
    fig.colorbar(sc2, ax=axes[2, :], label="Value")
    fig.colorbar(sc3, ax=axes[3, :], label="Value")
    axes[0, 0].set_ylabel(r"TraCS P($\tau_{k, i}$)", fontsize=15)
    axes[1, 0].set_ylabel(r"TraCS P($\tau_k$)", fontsize=15)
    axes[2, 0].set_ylabel(r"Backbone P($\tau_k$)", fontsize=15)
    axes[3, 0].set_ylabel(r"TraCS Guided P($\tau_k$)", fontsize=15)
    for i in range(6):
        axes[3, i].set_xlabel(rf"Traj. $\tau_{i}$", fontsize=15)

    fig.suptitle(rf"P($\tau_\text{{gt}}$) = {gt_score:.3f} for {scenario_id}")
    plt.show()


def compute_argoverse2_metrics(scored_predictions, data_root, scenario_id):
    # Backbone, TraCS alone, Backbone & TraCS
    KEYS = ["probability", "tracs_weight_normalized", "tracs_factor"]
    MAPPING = {"probability": "Backbone", "tracs_weight_normalized": "TraCS", "tracs_factor": "TraCS & Backbone"}

    # Get ground truth trajectory for comparison
    ground_truth = pd.read_parquet(data_root / "av2_mf_focal_test_annotations.parquet")
    gt_track = ground_truth[ground_truth["scenario_id"] == scenario_id].iloc[0]
    gt_track = np.array([gt_track["gt_trajectory_x"], gt_track["gt_trajectory_y"]]).T

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

    for key in KEYS:
        weights = scored_predictions[key].to_numpy()
        most_prob_index = weights.argmax()

        brier_ade = metrics.compute_brier_ade(predicted_tracks, gt_track, weights)
        brier_fde = metrics.compute_brier_fde(predicted_tracks, gt_track, weights)

        metrics_df_row = {
            "scenario_id": scenario_id,
            "track_id": scored_predictions.iloc[0]["track_id"],
            "key": MAPPING[key],
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

    return pd.DataFrame(rows)
