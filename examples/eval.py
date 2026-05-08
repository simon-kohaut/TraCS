# Standard Library
from pathlib import Path
import time

# Third Party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import av2.rendering.vector as vector_plotting_utils
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


def tracs_pointwise_to_score(pointwise_scores):
    N = len(pointwise_scores)
    return np.exp(np.sum(np.log(np.clip(pointwise_scores, 1e-8, 1.0))) / N)


def tracs_rate_traj(trajectory, logic, agent_loader, static_map, scenario_length: int = 60, write_speed: bool = True):
    # Decide dimensions of scneario
    x_traj = trajectory["x_traj"]
    y_traj = trajectory["y_traj"]
    xmin, xmax = x_traj.min(), x_traj.max()
    ymin, ymax = y_traj.min(), y_traj.max()

    # Initialize StaR Map and ProMis
    raster = CartesianRasterBand(
        origin=None,
        resolution=(1, 1),
        width=xmax - xmin + 50,
        height=ymax - ymin + 50,
    )   
    raster.data["east"] += (xmin + xmax) / 2
    raster.data["north"] += (ymin + ymax) / 2

    uam = CartesianMap(None, static_map.features + agent_loader.features)
    star_map = StaRMap(uam)
    star_map.initialize(
        evaluation_points=raster,
        number_of_random_maps=25,
        logic=logic,
    )
    promis = ProMis(star_map, logic, 1)
    promis.initialize(evaluation_points=raster, interpolation_method="nearest")
    if write_speed:
        speed_writer = promis._resin.make_writer_for("speed")

    num_timesteps = len(x_traj)
    agent_locations = CartesianCollection(None)
    agent_locations.append_with_default(np.zeros((1, 2)), 0.0)
    tracs_scores = []
    agent_loader.reset()
    for timestep in range(0, num_timesteps):
        # Update time
        agent_loader.update(1)

        # Get next predicted locations and transitions
        uam.features = static_map.features + agent_loader.features
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
        star_map.uam.features = static_map.features + agent_loader.features
        relations = star_map.get_all()
        for relation in relations:
            for location_type in relations[relation]:
                    star_map.update(relation, location_type, agent_locations, 10, "nearest")

        if write_speed:
            distance_traveled_km = np.linalg.norm(transition[0, :2] / 1000.0)
            time_in_hrs = 0.1 / 60 / 60
            speed = distance_traveled_km / time_in_hrs
            speed_writer.write(np.array([speed]), time.monotonic())

        # Get new inference results and append as tracs scores
        time.sleep(0.01)  # Ensures that on slow systems or large problems all data has already transferred
        updated_landscape = promis.update()
        if updated_landscape is not None:
            score = updated_landscape.values().item()
        
        tracs_scores.append(score)

    return np.array(tracs_scores)


def tracs_rate_scenario(
        scenario_id: str,
        logic: str, 
        prediction_path: Path,
        data_root: Path,
        mode: str,
        scenario_length: int = 60,
        trust: float = 1.0
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

    for prediction in agent_loader.focal_predictions:
        scores = tracs_rate_traj(prediction, logic, agent_loader, static_map, scenario_length)
        prediction["tracs_scores"] = scores.tolist()
        prediction["tracs_score"] = tracs_pointwise_to_score(scores)
        prediction["tracs_std"] = scores.std()
        prediction["tracs_weight"] = trust * prediction["tracs_score"] + (1 - trust) / 6

    ground_truth = pd.read_parquet(data_root / "av2_mf_focal_test_annotations.parquet")
    ground_truth_traj = ground_truth[ground_truth["scenario_id"] == scenario_id].iloc[0]
    ground_truth_traj["x_traj"] = ground_truth_traj["gt_trajectory_x"]
    ground_truth_traj["y_traj"] = ground_truth_traj["gt_trajectory_y"]
    gt_score = tracs_pointwise_to_score(tracs_rate_traj(ground_truth_traj, logic, agent_loader, static_map, scenario_length))

    scored_predictions = pd.DataFrame(agent_loader.focal_predictions)
    scored_predictions["tracs_weight_normalized"] = scored_predictions["tracs_weight"] / np.sum(scored_predictions["tracs_weight"])
    scored_predictions["tracs_factor"] = scored_predictions["probability"] * scored_predictions["tracs_weight"] \
        / np.sum(scored_predictions["probability"] * scored_predictions["tracs_weight"] ) # or use only / np.sum() to normalize?

    return scored_predictions, gt_score


def show_comparison_study(scored_predictions, gt_score, data_root, mode, scenario_id):
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
            axes[i, traj].set_aspect(1)
            axes[i, traj].set_xlim(xmin, xmax)        
            axes[i, traj].set_ylim(ymin, ymax)   

        sc0 = axes[0, traj].scatter(
            x_traj,
            y_traj,
            c=scored_predictions.iloc[traj]["tracs_scores"],
            cmap="cool_r", norm=tracs_scores_norm
        )
        sc1 = axes[1, traj].scatter(
            x_traj,
            y_traj,
            c=[scored_predictions.iloc[traj]["tracs_weight_normalized"]] * n_pts,
            cmap="cool_r", norm=tracs_norm
        )
        sc2 = axes[2, traj].scatter(
            x_traj,
            y_traj,
            c=[scored_predictions.iloc[traj]["probability"]] * n_pts,
            cmap="cool_r", norm=backbone_norm
        )
        sc3 = axes[3, traj].scatter(
            x_traj,
            y_traj,
            c=[scored_predictions.iloc[traj]["tracs_factor"]] * n_pts,
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
    fig.suptitle(rf"P($\tau_\text{{gt}}$) = {gt_score:.3f} for {scenario_id}")
    plt.show()
