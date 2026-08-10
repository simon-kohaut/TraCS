"""Core TraCS inference: scoring AV2 backbone trajectory predictions via ProMis/StaRMap."""

#
# Copyright (c) Simon Kohaut, Felix Divo, Julius Hahnewald, and contributors
#
# This file is part of TraCS and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with TraCS.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Standard Library
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Third Party
import numpy as np
import pandas as pd
from promis import ProMis, StaRMap
from promis.geo import CartesianCollection, CartesianMap, CartesianRasterBand

# TraCS
from tracs.av2.av2_loader import (
    AV2AgentLoader,
    AV2StaticMapLoader,
    load_focal_gt_trajectory,
)
from tracs.av2.metrics import tracs_pointwise_to_score

# Cache of compiled ProMis engines, keyed by (augmented_logic, dimension, max_models).
# AV2AgentLoader assigns agents canonical, identity-independent slot names (av2_loader.py's
# `slot_name`), so two different scenarios with the same agent-type-count profile compile to
# byte-identical Resin programs and can share an entry here, skipping Resin.compile() (subsantial
# part of per-scenario runtime). Reusing a compiled engine across scenarios is safe, as long as
# the reactive circuit's "current landscape" state is carried forward instead of reset; see the
# `_tracs_last_landscape` attribute stashed on the ProMis instance below.
_PROMIS_CACHE: dict[tuple[str, int, int | None], ProMis] = {}


def _tracs_rate_all(
    promis: ProMis,
    trajectories: list[dict[str, Any]],
    agent_loader: AV2AgentLoader,
    static_map: CartesianMap,
    speed_writer: Any | None = None,
    acceleration_writer: Any | None = None,
    yaw_rate_writer: Any | None = None,
    scenario_length: int = 60,
    sample_rate: float = 10,  # Argoverse 2 is sampled at 10 Hz
    number_of_random_maps: int = 10,
    callback_after_timestep: Callable[[int, float, CartesianCollection, ProMis], None] = lambda idx, time, landscape, promis: None,
) -> tuple[np.ndarray, float]:
    agent_loader.reset()
    # Initialize the variance with the minADE1 of DeMo
    agent_loader.apply_covariance({"all": np.power(1.639, 2) * np.eye(2)})

    num_timesteps = len(trajectories[0]["x_traj"])
    # The transition below repeats the last velocity by relying on `scenario_length`, so the
    # predicted horizon has to match it exactly; Argoverse 2 always gives 6 s at 10 Hz.
    assert num_timesteps == scenario_length, (
        f"Expected trajectories of {scenario_length} timesteps, got {num_timesteps}"
    )
    tracs_scores = np.zeros((len(trajectories), num_timesteps))
    last_speeds = None
    last_headings = None
    # Tracked separately so callers can report inference time net of this barrier;
    # see tracs_rate_scenario's `inference_seconds` attr.
    sleep_seconds = 0.0
    # When `promis` is a fresh (uncached) engine this is None, matching prior behavior.
    # When reused from _PROMIS_CACHE, seed from its last known landscape instead of
    # resetting to None: the reactive circuit may report "no change" on the very first
    # update after a scenario switch, and the correct value to report for an unchanged output
    # is exactly whatever the circuit's current state already is.
    landscape = getattr(promis, "_tracs_last_landscape", None)

    for timestep in range(num_timesteps):
        current_time = timestep / sample_rate

        # Get next predicted locations and transitions
        agent_loader.update(1)

        promis.star_map.uam.features = static_map.features | agent_loader.features
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
        promis.star_map.update(
            sample_points=agent_locations,
            number_of_random_maps=number_of_random_maps,
            interpolation_method="nearest",
            timestamp=current_time,
        )

        distance_traveled_km = np.linalg.norm(transitions / 1000.0, axis=1)
        time_in_s = 1.0 / sample_rate
        time_in_hrs = time_in_s / 60 / 60
        speed = distance_traveled_km / time_in_hrs  # km/h
        if last_speeds is None:
            last_speeds = speed  # Assume initial acceleration is zero
        acceleration = (speed - last_speeds) / time_in_s  # km/h/s
        last_speeds = speed

        heading = np.arctan2(transitions[:, 1], transitions[:, 0])  # rad
        if last_headings is None:
            last_headings = heading  # assume zero yaw_rate at t=0
        delta_heading = heading - last_headings
        delta_heading = (delta_heading + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
        yaw_rate = delta_heading / time_in_s  # rad/s
        last_headings = heading

        if speed_writer is not None:
            speed_writer.write(speed, current_time)
        if acceleration_writer is not None:
            acceleration_writer.write(acceleration, current_time)
        if yaw_rate_writer is not None:
            yaw_rate_writer.write(yaw_rate, current_time)

        # Get new inference results and append as tracs scores
        _sleep_start = time.perf_counter()
        time.sleep(0.01)  # Ensures that all data has already transferred
        sleep_seconds += time.perf_counter() - _sleep_start
        promis._evaluation_points = agent_locations
        updated_landscape = promis.update()
        if updated_landscape is not None:
            landscape = updated_landscape
            promis._tracs_last_landscape = landscape

        for i in range(len(trajectories)):
            tracs_scores[i, timestep] = landscape.values()[i, 0]

        # Call the callback function after each timestep
        callback_after_timestep(timestep, current_time, landscape, promis)

    return tracs_scores, sleep_seconds


def tracs_rate_scenario(
    scenario_id: str,
    logic: str,
    prediction_path: Path,
    data_root: Path,
    split: str,
    scenario_length: int = 60,
    max_models: int = 1_000_000,
    # Meant for custom inspection/visualization during the evaluation loop
    callback_after_timestep: Callable[[int, float, CartesianCollection, ProMis], None]
        # The callback receives the current index, time in seconds, landscape, and promis instance.
        = lambda idx, time, landscape, promis: None,
) -> tuple[pd.DataFrame, float]:
    """

    Notes:
        Speed (km/h), acceleration (km/h/s), and yaw_rate (rad/s) are written if requested as,
        e.g., `source("/sensors/yaw_rate", Number)`.
    """
    # Argoverse 2 scenario loading
    agent_loader = AV2AgentLoader(
        prediction_path=prediction_path,
        split_path=data_root / "av2" / split,
        scenario_id=scenario_id,
    )
    static_map_loader = AV2StaticMapLoader(
        path=data_root / "av2" / split / scenario_id / f"log_map_archive_{scenario_id}.json",
        focal_starting_location=agent_loader.focal_starting_location,
        focal_starting_speed=agent_loader.focal_starting_speed,
    )
    static_map = static_map_loader.to_cartesian_map()

    # Get ground truth trajectory and append to predictions for rating. Loaded here, before
    # engine setup, purely so `inference_seconds` below can be one contiguous span -- grading
    # (comparing against this ground truth) is not part of inference itself.
    gt_x, gt_y = load_focal_gt_trajectory(data_root, split, scenario_id)
    ground_truth_traj = {"x_traj": gt_x, "y_traj": gt_y}
    trajectories = agent_loader.focal_predictions + [ground_truth_traj]

    # Everything from here to the end of _tracs_rate_all is what a live deployment actually
    # pays for: compiling/reusing the inference engine and running the per-timestep reactive
    # loop. Scenario loading (above) and grading (below) are benchmark-only overhead.
    _inference_start = time.perf_counter()

    # Initialize StaR Map and ProMis
    raster = CartesianRasterBand(
        origin=None,
        resolution=(7, 1),
        width=1,
        height=1,
    )

    augmented_logic = logic + "\n# Appended automatically:\n" + agent_loader.resin_facts_and_sources
    # print("Augmented logic for ProMis:\n", augmented_logic)

    star_map = StaRMap(CartesianMap(None))
    star_map.initialize(
        evaluation_points=raster,
        number_of_random_maps=25,
        logic=augmented_logic,
    )

    promis_cache_key = (augmented_logic, 7, max_models)
    promis = _PROMIS_CACHE.get(promis_cache_key)
    if promis is None:
        promis = ProMis(star_map, augmented_logic, 7, verbose=False, max_models=max_models)
        _PROMIS_CACHE[promis_cache_key] = promis
    else:
        # Reuse the compiled Resin engine (the expensive part -- Resin.compile()); only
        # relink it to this scenario's own (cheap to build) StaRMap instance.
        promis.star_map = star_map
        star_map._promis = promis
    promis.initialize(evaluation_points=raster, interpolation_method="nearest")

    if 'source("/sensors/speed"' in augmented_logic:
        speed_writer = promis._resin.make_writer_for("speed")
    else:
        speed_writer = None
    if 'source("/sensors/acceleration"' in augmented_logic:
        acceleration_writer = promis._resin.make_writer_for("acceleration")
    else:
        acceleration_writer = None
    if 'source("/sensors/yaw_rate"' in augmented_logic:
        yaw_rate_writer = promis._resin.make_writer_for("yaw_rate")
    else:
        yaw_rate_writer = None

    # Score of TraCS alone
    scores, sleep_seconds = _tracs_rate_all(promis, trajectories, agent_loader, static_map,
                             speed_writer=speed_writer, acceleration_writer=acceleration_writer,
                             yaw_rate_writer=yaw_rate_writer,
                             scenario_length=scenario_length,
                             callback_after_timestep=callback_after_timestep)
    # Net of the sleep barrier (see _tracs_rate_all): time actually spent computing, not waiting.
    inference_seconds = (time.perf_counter() - _inference_start) - sleep_seconds

    for i, prediction in enumerate(agent_loader.focal_predictions):
        prediction["tracs_pointwise_scores"] = scores[i]
        prediction["tracs_trajectory_score"] = tracs_pointwise_to_score(scores[i])

    scored_predictions = pd.DataFrame(agent_loader.focal_predictions)
    # Normalize TraCS scores
    scored_predictions["tracs_trajectory_score"] /= np.sum(scored_predictions["tracs_trajectory_score"])
    # Rename backbone score for clarity
    scored_predictions["backbone_trajectory_score"] = scored_predictions["probability"]
    del scored_predictions["probability"]
    # Independent opinion pool (multiplicativ)
    scored_predictions["tracs_backbone_independent_op"] = scored_predictions["backbone_trajectory_score"] * scored_predictions["tracs_trajectory_score"]
    scored_predictions["tracs_backbone_independent_op"] /= np.sum(scored_predictions["tracs_backbone_independent_op"])
    # Linear opinion pool (additiv)
    scored_predictions["tracs_backbone_linear_op"] = scored_predictions["backbone_trajectory_score"] + scored_predictions["tracs_trajectory_score"]
    scored_predictions["tracs_backbone_linear_op"] /= np.sum(scored_predictions["tracs_backbone_linear_op"])

    # Deployment-relevant inference time (engine setup, cache-amortized, + the per-timestep
    # reactive loop), net of the sleep barrier -- excludes scenario loading and grading above.
    scored_predictions.attrs["inference_seconds"] = inference_seconds

    return scored_predictions, tracs_pointwise_to_score(scores[6])
