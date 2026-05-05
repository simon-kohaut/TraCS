
from argparse import ArgumentParser
from pathlib import Path
from typing import List
import multiprocessing
from functools import partial
from tqdm import tqdm
import time
import os

import pandas as pd
import numpy as np
from numpy.random import normal

from tracs.av2_loader import AV2AgentLoader, AV2StaticMapLoader
from promis import StaRMap, ProMis
from promis.geo import CartesianMap, CartesianRasterBand, CartesianCollection

N_WORKERS = 32
NUM_MAPS = 10

STEP_SIZE = 5

BUFFER_SPACE = 40
X_RES, Y_RES = 50, 50
RESOLUTION = X_RES * Y_RES

MODALITIES = 6
TRUST = 0.4

# TODO covariance

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
    "crossing": "PEDESTRIAN_CROSSING",
    "street": "DRIVABLE_AREA",
    "intersection": "INTERSECTION",
    "lane": "VEHICLE",
    "bike_lane": "BIKE",
    "bus_lane": "BUS",
    "solid_line": ["DOUBLE_SOLID_YELLOW", "DOUBLE_SOLID_WHITE", "SOLID_YELLOW", "SOLID_WHITE",],
    # "hybrid_line": ["DASH_SOLID_YELLOW", "DASH_SOLID_WHITE", "SOLID_DASH_WHITE", "SOLID_DASH_YELLOW",],
    "dashed_line": ["DASHED_WHITE", "DASHED_YELLOW", "DOUBLE_DASH_YELLOW", "DOUBLE_DASH_WHITE",],
} # should mostly be constant

def softmax(x):
    np.exp(x)/sum(np.exp(x))

def glob_files(data_root: Path, mode: str):
    file_root = data_root / mode
    scenario_files = list(file_root.rglob("*.parquet"))
    return scenario_files

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

    agent_loader = AV2AgentLoader(
        prediction_path=prediction_path,
        split_path=data_root / mode / "raw",
        scenario_id=scenario_id,
    )

    agent_loader.reset()

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
    for timestep in range(0, num_timesteps, STEP_SIZE): 
        # score first self.update_frequency positions on one star map 
        landscape = promis.update()
        if landscape is not None:
            interpolator = landscape.get_interpolator()

        for prediction in agent_loader.focal_predictions:
            coords = np.array([
                prediction["x_traj"][timestep:timestep+STEP_SIZE],
                prediction["y_traj"][timestep:timestep+STEP_SIZE],
            ])
            tracs_scores: list = prediction.get("tracs_scores", [])
            tracs_scores.extend(interpolator(coords.T))
            prediction["tracs_scores"] = tracs_scores

        agent_loader.update(STEP_SIZE)
        uam.features = static_map.features + agent_loader.features
        agent_locations = CartesianCollection(None)
        agent_locations.append_with_default(
            np.vstack([normal(a.to_numpy().T, 15.0, [50, 2]) for a in agent_loader.features]), 0.0
        )
        star_map.update("distance", "vehicle", agent_locations, 25) # TODO make less specific

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



def preprocess(args):
    data_root = Path(args.data_root)

    for mode in ["test"]:
        os.makedirs(data_root / "tracs_predictions" / mode, exist_ok=True)
        
        mode_dir = data_root / mode / "raw"
        scenario_files = os.listdir(mode_dir)[::100]

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
    # parser.add_argument("--batch", "-b", type=int, default=50)
    parser.add_argument("--parallel", "-p", action="store_true")
    parser.add_argument("--initial_predictions", "-i", type=str, default=None)

    args = parser.parse_args()

    preprocess(args)
