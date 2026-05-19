from argparse import ArgumentParser
from pathlib import Path
import multiprocessing
from functools import partial
from tqdm import tqdm
import time
import os

import pandas as pd
import numpy as np

from tracs.av2_loader import AV2AgentLoader, AV2StaticMapLoader
from promis import StaRMap, ProMis
from promis.geo import CartesianMap, CartesianRasterBand, CartesianCollection

N_WORKERS = 75
NUM_MAPS = 20

STEP_SIZE = 1
SCENARIO_LENGTH = 60

BUFFER_SPACE = 20
X_RES, Y_RES = 50, 50
RESOLUTION = X_RES * Y_RES

MODALITIES = 6
TRUST = 0.4

tracs_logic = """
distance(vehicle) <- source("/star_map/distance/vehicle", Density).
speed <- source("/sensors/speed", Number).

# Define three speed levels for city driving with corresponding safe distances
low_speed if speed < 20.
medium_speed if speed > 20 and speed < 40.
high_speed if speed > 40.

# Safe distance requirements for each speed level
safe_at_low if low_speed and distance(vehicle) > 5.
safe_at_medium if medium_speed and distance(vehicle) > 10.
safe_at_high if high_speed and distance(vehicle) > 20.

# Landscape: safe distance must be maintained according to speed level
landscape if safe_at_low.
landscape if safe_at_medium.
landscape if safe_at_high.

landscape -> target("/landscape").
""" # TODO take from json


FEATURE_DESCRIPTION = {
    "crosswalk": "PEDESTRIAN_CROSSING",
    "drivable_area": "DRIVABLE_AREA",
    "intersection": "INTERSECTION",
    "vehicle_lane": "VEHICLE",
    "bike_lane": "BIKE",
    "bus_lane": "BUS",
    "solid_line": ["DOUBLE_SOLID_YELLOW", "DOUBLE_SOLID_WHITE", "SOLID_YELLOW", "SOLID_WHITE",],
    "dashed_line": ["DASHED_WHITE", "DASHED_YELLOW", "DOUBLE_DASH_YELLOW", "DOUBLE_DASH_WHITE",],
} # according to system prompt


PREAMBLE_STATIC = """
    distance(intersection) <- source("/star_map/distance/intersection", Density).
    distance(crosswalk) <- source("/star_map/distance/crosswalk", Density).
    distance(vehicle_lane) <- source("/star_map/distance/vehicle_lane", Density).
    distance(bike_lane) <- source("/star_map/distance/bike_lane", Density).
    distance(bus_lane) <- source("/star_map/distance/bus_lane", Density).
    
    over(drivable_area) <- source("/star_map/over/drivable_area", Probability).
    over(vehicle_lane) <- source("/star_map/over/vehicle_lane", Probability).
    over(bike_lane) <- source("/star_map/over/bike_lane", Probability).
    over(bus_lane) <- source("/star_map/over/bus_lane", Probability).
    over(crosswalk) <- source("/star_map/over/crosswalk", Probability).
    over(intersection) <- source("/star_map/over/intersection", Probability).
""" 

PREAMBLE_DYNAMIC = """
    distance(vehicle) <- source("/star_map/distance/vehicle", Density).
    distance(pedestrian) <- source("/star_map/distance/pedestrian", Density).
    distance(bus) <- source("/star_map/distance/bus", Density).
    distance(bike) <- source("/star_map/distance/bike", Density).
    distance(motorcycle) <- source("/star_map/distance/intersection", Density).

    opposes(vehicle_lane_centerline) <- source("/star_map/opposes/vehicle_lane_centerline", Probability).
    opposes(bike_lane_centerline) <- source("/star_map/opposes/bike_lane_centerline", Probability).
    opposes(bus_lane_centerline) <- source("/star_map/opposes/bus_lane_centerline", Probability).

    follows(vehicle_lane_centerline) <- source("/star_map/follows/vehicle_lane_centerline", Probability).
    follows(bike_lane_centerline) <- source("/star_map/follows/bike_lane_centerline", Probability).
    follows(bus_lane_centerline) <- source("/star_map/follows/bus_lane_centerline", Probability).
    
    intersects(dashed_line) <- source("/star_map/intersects/dashed_line", Probability).
    intersects(solid_line) <- source("/star_map/intersects/solid_line", Probability).
    intersects(crosswalk) <- source("/star_map/intersects/crosswalk", Probability).

    enters(intersection) <- source("/star_map/enters/intersection", Probability).
    enters(crosswalk) <- source("/star_map/enters/crosswalk", Probability).
    enters(bike_lane) <- source("/star_map/enters/bike_lane", Probability).
    enters(bus_lane) <- source("/star_map/enters/bus_lane", Probability).
    enters(vehicle_lane) <- source("/star_map/enters/vehicle_lane", Probability).

    exits(intersection) <- source("/star_map/exits/intersection", Probability).
    exits(crosswalk) <- source("/star_map/exits/crosswalk", Probability).
    exits(bike_lane) <- source("/star_map/exits/bike_lane", Probability).
    exits(bus_lane) <- source("/star_map/exits/bus_lane", Probability).
    exits(vehicle_lane) <- source("/star_map/exits/vehicle_lane", Probability).

    speed <- source("/sensors/speed", Number).
"""

def tracs_pipeline(
        scenario_id: str,
        logic: str, 
        feature_description: dict[str, str | list[str]],
        prediction_path: Path,
        data_root: Path,
        mode: str,
        write: bool = True,
    ) -> None:

    static_map = AV2StaticMapLoader(
        path=data_root / mode / "raw" / scenario_id / f"log_map_archive_{scenario_id}.json",
        feature_description=feature_description,
    ).to_cartesian_map()
    static_map.apply_covariance(0.2 * np.eye(2))

    agent_loader = AV2AgentLoader(
        prediction_path=prediction_path,
        split_path=data_root / mode / "raw",
        scenario_id=scenario_id,
    )

    agent_loader.reset()
    agent_loader.apply_covariance({"all": np.power(1.639, 2) * np.eye(2)}) #minADE1 DeMo

    uam = CartesianMap(None, static_map.features + agent_loader.features)
    star_map = StaRMap(uam)

    x_pred_flat = np.array([f_pred["x_traj"] for f_pred in agent_loader.focal_predictions]).ravel()
    y_pred_flat = np.array([f_pred["y_traj"] for f_pred in agent_loader.focal_predictions]).ravel()
    xmin, xmax = x_pred_flat.min(), x_pred_flat.max()
    ymin, ymax = y_pred_flat.min(), y_pred_flat.max()

    raster = CartesianRasterBand(
        origin=None,
        resolution=(X_RES, Y_RES),
        width=xmax - xmin + BUFFER_SPACE,
        height=ymax - ymin + BUFFER_SPACE,
    )   
    raster.data["east"] += (xmin + xmax) / 2
    raster.data["north"] += (ymin + ymax) / 2

    star_map.initialize(
        evaluation_points=raster,
        number_of_random_maps=NUM_MAPS,
        logic=logic,
    )

    promis = ProMis(star_map, logic, RESOLUTION)
    writer = promis._resin.make_writer_for("speed")
    writer.write(np.array([30] * RESOLUTION), time.monotonic()) # TODO integrate smart speed
    promis.initialize(
        evaluation_points=raster,
    )

    star_map.link(promis)

    rc = promis.get_reactive_circuit()    

    # lift every distance relation, as they all change here. 
    # TODO if in composite scenario, then maybe add furter constraints
    names = promis.get_names()
    dynamic_relation_indices = [i for i, name in enumerate(names) if "distance_" in name]
    for i in dynamic_relation_indices:
        rc.lift_leaf(i)
        rc.lift_leaf(i + 1)

    num_timesteps = len(agent_loader.focal_predictions[0]["x_traj"])

    # TODO add gt evaluation here already?

    interpolator: callable
    for timestep in range(0, num_timesteps-1): 
        # score first self.update_frequency positions on one star map 
        landscape = promis.update()
        if landscape is not None:
            interpolator = landscape.get_interpolator()

        for prediction in agent_loader.focal_predictions:
            coords = np.array([
                prediction["x_traj"][timestep],
                prediction["y_traj"][timestep],
            ])
            tracs_scores: list = prediction.get("tracs_scores", [])
            tracs_scores.extend(interpolator(coords.T))
            prediction["tracs_scores"] = tracs_scores

        agent_loader.update(STEP_SIZE)
        uam.features = agent_loader.features
        agent_locations = CartesianCollection(None)
        positions = np.vstack([np.array([a["x_traj"], a["y_traj"]])[:, timestep] for a in agent_loader.focal_predictions])
        if timestep < SCENARIO_LENGTH - 1:
            next_positions = np.vstack([np.array([a["x_traj"], a["y_traj"]])[:, timestep+1] for a in agent_loader.focal_predictions])
            transitions=np.hstack((next_positions, np.zeros((len(agent_loader.focal_predictions), 1))))
        
        agent_locations.append_with_default(
            coordinates=positions,
            value=0.0,
            transitions=transitions,
        )
        star_map.update("distance", "vehicle", agent_locations, 2) # TODO make less specific
        # TODO update only on next steps per trajectories, not entire raster
        # for the starmap update, only features mentioned in the update are necessary. 

    # post processing
    for prediction in agent_loader.focal_predictions:
        scores = np.array(prediction["tracs_scores"])
        prediction["tracs_scores"] = scores.tolist()
        prediction["tracs_score"] = scores.mean()
        prediction["tracs_std"] = scores.std()
        prediction["tracs_weight"] = TRUST * prediction["tracs_score"] + (1 - TRUST) / MODALITIES
        
    scored_predictions = pd.DataFrame(agent_loader.focal_predictions)
    scored_predictions["tracs_weight_normalized"] = scored_predictions["tracs_weight"] / np.sum(scored_predictions["tracs_weight"])
    scored_predictions["tracs_factor"] = scored_predictions["probability"] * scored_predictions["tracs_weight"] \
        / np.sum(scored_predictions["probability"] * scored_predictions["tracs_weight"] ) # or use only / np.sum() to normalize?

    if write:
        scored_predictions.to_parquet(data_root / "tracs_predictions" / mode / f"{scenario_id}.parquet")
        return None
    else: 
        return scored_predictions


def _cache_star_maps(
        scenario_id: str,
        feature_description: dict[str, str | list[str]],
        prediction_path: Path,
        data_root: Path,
        mode: str,
    ) -> None:

    static_map = AV2StaticMapLoader(
        path=data_root / mode / "raw" / scenario_id / f"log_map_archive_{scenario_id}.json",
        feature_description=feature_description,
    ).to_cartesian_map()
    static_map.apply_covariance(0.0025 * np.eye(2))

    agent_loader = AV2AgentLoader(
        prediction_path=prediction_path,
        split_path=data_root / mode / "raw",
        scenario_id=scenario_id,
    )

    agent_loader.reset()
    agent_loader.apply_covariance({"all": np.power(1.639, 2) * np.eye(2)}) #minADE1 DeMo

    uam = CartesianMap(None, static_map.features + agent_loader.features)
    star_map = StaRMap(uam)

    x_pred_flat = np.array([f_pred["x_traj"] for f_pred in agent_loader.focal_predictions]).ravel()
    y_pred_flat = np.array([f_pred["y_traj"] for f_pred in agent_loader.focal_predictions]).ravel()
    xmin, xmax = x_pred_flat.min(), x_pred_flat.max()
    ymin, ymax = y_pred_flat.min(), y_pred_flat.max()

    raster = CartesianRasterBand(
        origin=None,
        resolution= (X_RES, Y_RES), #(10, 10),
        width=xmax - xmin + BUFFER_SPACE,
        height=ymax - ymin + BUFFER_SPACE,
    )   
    raster.data["east"] += (xmin + xmax) / 2
    raster.data["north"] += (ymin + ymax) / 2

    star_map.initialize(
        evaluation_points=raster,
        number_of_random_maps=NUM_MAPS,
        logic=PREAMBLE_STATIC,
    )

    # def candidate_sampler():
    #     collection = CartesianCollection.make_latin_hypercube(None, raster.width, raster.height, number_of_samples=100, include_corners=True)
    #     collection.data["east"] += (xmin + xmax) / 2
    #     collection.data["north"] += (ymin + ymax) / 2
    #     return collection

    # star_map.adaptive_sample(
    #     candidate_sampler=candidate_sampler,
    #     number_of_random_maps=20,
    #     number_of_iterations=10,
    #     number_of_improvement_points=25,
    #     acquisition_method = "entropy",
    #     scaler=5,
    # )

    star_map.save(data_root / "star_maps" / f"{scenario_id}.pkl")


def main(args):
    data_root = Path(args.data_root)

    for mode in ["test"]:
        os.makedirs(data_root / "tracs_predictions" / mode, exist_ok=True)
        
        mode_dir = data_root / mode / "raw"
        scenario_files = os.listdir(mode_dir)

        if args.cache: 
            os.makedirs(data_root / "star_maps" / mode, exist_ok=True)

            cache_star = partial(
                _cache_star_maps,
                # scenario_id remains as a variable argument 
                feature_description=FEATURE_DESCRIPTION,
                prediction_path=args.initial_predictions,
                data_root=data_root,
                mode=mode,
            )

            if args.parallel:
                with multiprocessing.Pool(N_WORKERS) as p:
                    for _ in tqdm(p.imap(cache_star, scenario_files), total=len(scenario_files)):
                        pass
            else:
                for file in tqdm(scenario_files):
                    print(file)
                    cache_star(file)

        if not args.cache: 
            pipeline_func = partial(
                tracs_pipeline,
                logic=tracs_logic,
                feature_description=FEATURE_DESCRIPTION,
                prediction_path=args.initial_predictions,
                # scenario_id remains as a variable argument 
                data_root=data_root,
                mode=mode,
                write=True,
            )

            if args.parallel:
                with multiprocessing.Pool(N_WORKERS) as p:
                    for _ in tqdm(p.imap(pipeline_func, scenario_files), total=len(scenario_files)):
                        pass
            else:
                for file in tqdm(scenario_files):
                    pipeline_func(file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", "-d", type=str, default='/path/to/data_root')
    parser.add_argument("--parallel", "-p", action="store_true")
    parser.add_argument("--cache", "-c", action="store_true")
    parser.add_argument("--initial_predictions", "-i", type=str, default=None)

    args = parser.parse_args()

    main(args)
