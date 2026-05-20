"""Loaders for fetching spatial data from static AV2 Maps."""

#
# Copyright (c) Simon Kohaut, Felix Divo, Julius Hahnewald, and contributors
#
# This file is part of TraCS and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with TraCS.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Standard Library
from functools import cached_property
from collections import defaultdict
from pathlib import Path

# Third Party
from av2.map.map_api import ArgoverseStaticMap
import pandas as pd
import numpy as np

# ProMis
from promis.geo import CartesianLocation, CartesianPolygon, CartesianPolyLine, CartesianMap
from promis.loaders.spatial_loader import SpatialLoader


class AV2StaticMapLoader(SpatialLoader):
    """A loader for spatial data from the Argoverse 2 (AV2) dataset.

    Extracts features directly from an AV2 static map into a CartesianMap with an unknown, city-specific origin.
    Values that can be referenced in a feature description are 
        "VEHICLE", "BUS", "BIKE"; 
        "INTERSECTION";
        "DRIVABLE_AREA";  
        "PEDESTRIAN_CROSSING";
    for describing different spaces on the map. For lane markings, the following can be used:
        "DASH_SOLID_YELLOW", "DASH_SOLID_WHITE", "DASHED_WHITE", "DASHED_YELLOW", 
        "DOUBLE_SOLID_YELLOW", "DOUBLE_SOLID_WHITE", "DOUBLE_DASH_YELLOW", "DOUBLE_DASH_WHITE",
        "SOLID_YELLOW", "SOLID_WHITE", "SOLID_DASH_WHITE", "SOLID_DASH_YELLOW", "SOLID_BLUE", "NONE", "UNKNOWN".

    Note:
        If a `feature_description` is provided during initialization, data loading via the
        `load` method is triggered immediately. If it is not provided, you must call
        `load()` manually with a feature description to populate the loader with data.

    Args:
        path: Path to the AV2 map JSON file.
        feature_description: A mapping from a location type (e.g., "drivable_area") to
            an AV2 map feature type (e.g., "drivable_area", "lane_segment", "pedestrian_crossing").
            If provided, data is loaded upon initialization. Defaults to None.
    """

    def __init__(
        self,
        path: str | Path,
        feature_description: dict[str, str | list[str]] | None = None,
    ):
        self.path = Path(path)
        self.features = defaultdict(list, {})

        # Load the Argoverse 2 map API
        self.av2_map = ArgoverseStaticMap.from_json(self.path)
        self.origin = None
        if feature_description is not None:
            self.load(feature_description)

    def load(self, feature_description: dict[str, str | list[str]]) -> None:
        """Load features from the AV2 map that match the given descriptions.

        Supported AV2 map features:
            - drivable_area
            - pedestrian_crossing
            - lane_segment

        Args:
            feature_description: A mapping from a location type to an AV2 feature
                name (or list of them).
        """

        def invert_feature_description(
            feature_description: dict[str, str | list[str]]
        ) -> dict[str, str]:
            inv = {}
            for k, v in feature_description.items():
                if isinstance(v, list):
                    inv |= {item: k for item in v}
                else:
                    inv[v] = k
            return inv

        inverted_fd = invert_feature_description(feature_description)
        self._load_drivable_areas(inverted_fd)
        self._load_pedestrian_crossings(inverted_fd)
        self._load_lane_segments(inverted_fd)


    def _load_drivable_areas(self, inverted_fd: dict[str, str]) -> None:
        location_type = inverted_fd["DRIVABLE_AREA"]
        for da in self.av2_map.vector_drivable_areas.values():
            coords = [CartesianLocation(east=pt[0], north=pt[1]) for pt in da.xyz]
            self.features[location_type].append(CartesianPolygon(coords, location_type=location_type))

    def _load_pedestrian_crossings(self, inverted_fd: dict[str, str]) -> None:
        location_type = inverted_fd["PEDESTRIAN_CROSSING"]
        for pc in self.av2_map.vector_pedestrian_crossings.values():
            coords = [CartesianLocation(east=pt[0], north=pt[1]) for pt in pc.polygon]
            self.features[location_type].append(CartesianPolygon(coords, location_type=location_type))

    def _load_lane_segments(self, inverted_fd: dict[str, str]) -> None:
        for ls in self.av2_map.vector_lane_segments.values():
            lane_type = inverted_fd.get(ls.lane_type.value, str(ls.lane_type.value))

            coords = [CartesianLocation(east=pt[0], north=pt[1]) for pt in ls.polygon_boundary]
            self.features[lane_type].append(CartesianPolygon(coords, location_type=lane_type))
            if ls.is_intersection:
                location_type = inverted_fd.get("INTERSECTION", "INTERSECTION")
                self.features[location_type].append(CartesianPolygon(coords, location_type=location_type))

            centerline = self.av2_map.get_lane_segment_centerline(ls.id)
            center_coords = [CartesianLocation(east=pt[0], north=pt[1]) for pt in centerline]
            location_type = f"{lane_type}_centerline"
            self.features[location_type].append(CartesianPolyLine(center_coords, location_type=location_type))

            for lane_marking in [ls.left_lane_marking, ls.right_lane_marking]:
                lane_marking_type = inverted_fd.get(lane_marking.mark_type.value, str(lane_marking.mark_type.value))
                coords = [CartesianLocation(east=pt[0], north=pt[1]) for pt in lane_marking.polyline]
                self.features[lane_marking_type].append(CartesianPolyLine(coords, location_type=lane_marking_type))

    def to_cartesian_map(self) -> CartesianMap:
        """Creates a CartesianMap from the loaded features.

        Returns:
            A CartesianMap containing all loaded features, with no known origin.
        """
        return CartesianMap(None, self.features)


class AV2AgentLoader(SpatialLoader):
    """A loader for agent trajectories from a parquet file for a specific scenario.

    Note:
        We assume that an agent type never changes over time.
    
    Args:
        threshold_is_moving: A threshold for determining whether an agent is considered "moving" based on its predicted trajectory.
            Moving iff avgerage speed is more than this threshold in m/s. This is used to generate "moving(agent_X)." facts for the logic.
    """

    def __init__(self,
        prediction_path: str | Path,
        split_path: str | Path,
        scenario_id: str,
        include_individual_agents: bool = True,
        threshold_is_moving: float = 0.25,
    ):
        self.prediction_path = Path(prediction_path)
        self.scenario_path = Path(split_path) / scenario_id / f"scenario_{scenario_id}.parquet"
        self.scenario_id = scenario_id
        self.features = defaultdict(list, {})
        self.focal_predictions = []
        self.agents = []  # This exludes the focal agent, which is stored separately in `focal_predictions`
        self.current_index = 0
        self.include_individual_agents = include_individual_agents
        self.threshold_is_moving = threshold_is_moving

        self.load()

    def apply_covariance(self, covariance: dict[str, float]) -> None:
        """
        Set covariance for the agent features. If "all" is in covariance, use only that.
        
        Args: 
            covariance: dict[str, float], mapping vehicle type to covariance matrix.
        """

        if "all" in covariance:
            for agent in self.agents:
                agent["covariance"] = covariance["all"]
        else:
            for agent in self.agents:
                agent["covariance"] = covariance.get(agent["agent_type"], 1)

    def load(self) -> None:
        """Load predicted trajectories for the specified scenario from the prediction parquet file."""

        setup = pd.read_parquet(self.scenario_path)
        focal_track_id = setup["focal_track_id"].iloc[0]

        df = pd.read_parquet(self.prediction_path, filters=[("scenario_id", "==", self.scenario_id)])
        # For simplicity, we only take the best non-focal predictions
        # Note that the current focal agent must not be the "AV" (Autonomous Vehicle) track,
        # since we also evaluate on other agents in the scenario.
        focal_mask = df["track_id"] == focal_track_id
        best_non_focal = df[~focal_mask].sort_values("probability", ascending=False).drop_duplicates("track_id", keep="first")
        df = pd.concat([df[focal_mask], best_non_focal])

        for _, row in df.iterrows():
            track_id = row.get("track_id", "unknown")
            prob = row.get("probability", 1.0)

            x_traj = row.get("predicted_trajectory_x", [])
            y_traj = row.get("predicted_trajectory_y", [])
            assert len(x_traj) == len(y_traj), f"Trajectory length mismatch for track_id {track_id} in scenario {self.scenario_id}."
            agent_type = setup[setup["track_id"] == track_id]["object_type"].iloc[0]

            if len(x_traj) > 0 and len(y_traj) > 0:
                # Calculate average speed over the predicted trajectory to determine if the agent is moving
                distances = np.sqrt(np.diff(x_traj)**2 + np.diff(y_traj)**2)
                total_distance = np.sum(distances)
                total_time = len(x_traj) * 0.1  # Assuming predictions are at 10 Hz
                avg_speed = total_distance / total_time
                is_moving = avg_speed > self.threshold_is_moving
            else:
                is_moving = False
                print(f"Warning: No trajectory data for track_id {track_id} in scenario {self.scenario_id}.")

            state = {
                "track_id": track_id,
                "probability": prob,
                "x_traj": x_traj,
                "y_traj": y_traj,
                "agent_type": agent_type,
                "covariance": 1,
                "is_moving": is_moving,
            }

            if track_id == focal_track_id:
                self.focal_predictions.append(state)
            else:
                self.agents.append(state)

        self._build_features()

    def update(self, step_size = 1) -> None:
        """Advance to the next index in the trajectories and update features.

        Args:
            step_size: How many frames to skip in one step.
        """
        self.current_index += step_size
        self._build_features()

    def _build_features(self) -> None:
        """Update the features list to represent the agents at the current index."""
        self.features = defaultdict(list, {})
        agent_states = self.agents # [self.focal_predictions] + self.agents
        for agent in agent_states:
            x_traj = agent["x_traj"]
            y_traj = agent["y_traj"]

            index = min(self.current_index, len(x_traj)-1)

            x = float(x_traj[index])
            y = float(y_traj[index])
            # TODO: incorporate agent["probability"] for certainty in prediction (dropped in load() for simplicity for now)

            location = CartesianLocation(
                east=x, north=y, 
                name=agent['track_id'], 
                location_type=agent["agent_type"], 
                covariance=agent["covariance"] * np.eye(2),
            )
            self.features[agent["agent_type"]].append(location)
            if self.include_individual_agents:
                self.features[f"agent_{agent['track_id']}"].append(location)

    def to_cartesian_map(self) -> CartesianMap:
        """Creates a CartesianMap from the loaded agent features.
        
        Returns:
            A CartesianMap containing all agents at the current time.
        """
        return CartesianMap(None, self.features)
    
    def reset(self) -> None:
        """Reset the agents to their initial positions."""
        self.current_index = 0
        self._build_features()

    @cached_property
    def resin_facts_and_sources(self) -> str:
        """A string of Resin facts describing the focal agent and the non-focal agents."""
        facts = ""

        # One of "is_vehicle.", "is_bus.", "is_cyclist.", "is_motorcyclist." or "is_pedestrian."
        facts += f"is_{self.focal_predictions[0]['agent_type']}.\n"

        if self.include_individual_agents:
            for agent in self.agents:
                agent_name = f"agent_{agent['track_id']}"

                # Facts like "vehicle(agent_123).", "pedestrian(agent_456).", etc.
                facts += f"{agent['agent_type']}({agent_name}).\n"

                facts += f"distance({agent_name}) <- source(\"/star_map/distance/{agent_name}\", Density).\n"

                # Facts like "moving(agent_123)." if the agent is predicted to be non-stationary (e.g., not parked)
                if agent.get('is_moving', False):
                    facts += f"moving({agent_name}).\n"

        return facts
