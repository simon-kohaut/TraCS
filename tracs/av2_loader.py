"""Loaders for fetching spatial data from static AV2 Maps."""

#
# Copyright (c) Simon Kohaut, Felix Divo, Julius Hahnewald, and contributors
#
# This file is part of TraCS and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with TraCS.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Standard Library
import math
from functools import cached_property
from collections import defaultdict, deque
from pathlib import Path

# Third Party
from av2.geometry.utm import CityName, CITY_ORIGIN_LATLONG_DICT
from av2.map.map_api import ArgoverseStaticMap
import pandas as pd
import numpy as np

# ProMis
from promis.geo import CartesianLocation, CartesianPolygon, CartesianPolyLine, CartesianMap, PolarLocation
from promis.loaders.spatial_loader import SpatialLoader


CITY_NAME_MAP = {
    "austin": CityName.ATX,
    "miami": CityName.MIA,
    "pittsburgh": CityName.PIT,
    "washington-dc": CityName.WDC,
    "palo-alto": CityName.PAO,
    "dearborn": CityName.DTW,
}

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


def city_to_origin(city_name: str) -> PolarLocation:
    lat, lon = CITY_ORIGIN_LATLONG_DICT[CITY_NAME_MAP[city_name]]
    return PolarLocation(longitude=lon, latitude=lat)


def load_focal_gt_trajectory(data_root: Path, split: str, scenario_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the focal track's ground-truth future trajectory as (x, y) numpy arrays.

    For test split the GT lives in the separate annotations parquet; for val/train it
    lives inside the scenario parquet as rows where `observed` is False.
    """
    if split == "test":
        ann = pd.read_parquet(
            data_root / "av2" / "av2_mf_focal_test_annotations.parquet",
            filters=[("scenario_id", "==", scenario_id)],
        )
        row = ann.iloc[0]
        return np.asarray(row["gt_trajectory_x"]), np.asarray(row["gt_trajectory_y"])
    scenario = pd.read_parquet(data_root / "av2" / split / scenario_id / f"scenario_{scenario_id}.parquet")
    focal_id = scenario["focal_track_id"].iloc[0]
    future = scenario[(scenario["track_id"] == focal_id) & (~scenario["observed"])].sort_values("timestep")
    return future["position_x"].to_numpy(), future["position_y"].to_numpy()


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
        focal_starting_location: An optional CartesianLocation representing the starting location of the focal agent.
            Used for extracting the "focal_lane" feature, which includes all lane segments connected to the lane segment closest to this location.
        threshold_focal_lane: Determines whether an agent is considered to be on the focal lane. Distance to the centerline in meters.
        max_focal_lane_hops: Maximum number of lane-graph hops to explore from the focal lane in each direction
            (forward via successors and backward via predecessors). Use ``math.inf`` to disable the limit.
    """

    def __init__(
        self,
        path: str | Path,
        feature_description: dict[str, str | list[str]] | None = FEATURE_DESCRIPTION,
        focal_starting_location: CartesianLocation | None = None,
        threshold_focal_lane: float = 1.0,
        max_focal_lane_hops: float = math.inf,
    ):
        self.path = Path(path)
        self.features = defaultdict(list, {})

        # Load the Argoverse 2 map API
        self.av2_map = ArgoverseStaticMap.from_json(self.path)
        self.origin = None
        self.focal_starting_location = focal_starting_location
        self.threshold_focal_lane = threshold_focal_lane
        self.max_focal_lane_hops = max_focal_lane_hops

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

        inverted_fd = {}
        for k, v in feature_description.items():
            if isinstance(v, list):
                inverted_fd |= {item: k for item in v}
            else:
                inverted_fd[v] = k

        self._load_drivable_areas(inverted_fd)
        self._load_pedestrian_crossings(inverted_fd)
        self._load_lane_segments(inverted_fd)

    def _load_drivable_areas(self, inverted_fd: dict[str, str]) -> None:
        location_type = inverted_fd["DRIVABLE_AREA"]
        for da in self.av2_map.vector_drivable_areas.values():
            coords = [CartesianLocation(east=pt[0], north=pt[1]) for pt in da.xyz]
            self.features[location_type].append(CartesianPolygon(coords, name=da.id, location_type=location_type))

    def _load_pedestrian_crossings(self, inverted_fd: dict[str, str]) -> None:
        location_type = inverted_fd["PEDESTRIAN_CROSSING"]
        for pc in self.av2_map.vector_pedestrian_crossings.values():
            coords = [CartesianLocation(east=pt[0], north=pt[1]) for pt in pc.polygon]
            self.features[location_type].append(CartesianPolygon(coords, name=pc.id, location_type=location_type))

    def _load_lane_segments(self, inverted_fd: dict[str, str]) -> None:
        lane_centerlines: list[CartesianPolyLine] = []
        for ls in self.av2_map.vector_lane_segments.values():
            lane_type = inverted_fd.get(ls.lane_type.value, str(ls.lane_type.value))

            coords = [CartesianLocation(east=pt[0], north=pt[1]) for pt in ls.polygon_boundary]
            self.features[lane_type].append(CartesianPolygon(coords, name=ls.id, location_type=lane_type))
            if ls.is_intersection:
                location_type = inverted_fd.get("INTERSECTION", "INTERSECTION")
                self.features[location_type].append(CartesianPolygon(coords, name=ls.id, location_type=location_type))

            centerline = self.av2_map.get_lane_segment_centerline(ls.id)
            center_coords = [CartesianLocation(east=pt[0], north=pt[1]) for pt in centerline]
            location_type = f"{lane_type}_centerline"
            centerline_polyline = CartesianPolyLine(center_coords, name=ls.id, location_type=location_type)
            lane_centerlines.append(centerline_polyline)
            self.features[location_type].append(centerline_polyline)

            for direction, lane_marking in [("left", ls.left_lane_marking), ("right", ls.right_lane_marking)]:
                lane_marking_type = inverted_fd.get(lane_marking.mark_type.value, str(lane_marking.mark_type.value))
                coords = [CartesianLocation(east=pt[0], north=pt[1]) for pt in lane_marking.polyline]
                name = f"{direction}_{lane_marking_type}_{ls.id}"
                self.features[lane_marking_type].append(CartesianPolyLine(coords, name=name, location_type=lane_marking_type))

        self._load_focal_lane(lane_centerlines)

    def _load_focal_lane(self, lane_centerlines: list[CartesianPolyLine]) -> None:
        if self.focal_starting_location is None:
            return

        # Find the lane centerline closest to the focal starting location
        closest_centerline = None
        min_dist = float("inf")
        for centerline in lane_centerlines:
            dist = centerline.distance(self.focal_starting_location)
            if dist < min_dist:
                min_dist = dist
                closest_centerline = centerline

        if closest_centerline is None or min_dist > self.threshold_focal_lane:
            print(f"No suitable focal lane found within {self.threshold_focal_lane}m.")
            return

        focal_id = int(closest_centerline.name)

        # Build the complete lane connectivity graph in both directions
        successors: dict[int, list[int]] = defaultdict(list)
        predecessors: dict[int, list[int]] = defaultdict(list)
        for ls in self.av2_map.vector_lane_segments.values():
            successors[ls.id] = self.av2_map.get_lane_segment_successor_ids(ls.id) or []
            for succ_id in successors[ls.id]:
                predecessors[succ_id].append(ls.id)

        # BFS up to max_focal_lane_hops in each direction from the focal lane
        connected_lane_ids: set[int] = {focal_id}
        for neighbors in (successors, predecessors):
            queue: deque[tuple[int, int]] = deque([(focal_id, 0)])
            while queue:
                current_id, hops = queue.popleft()
                if hops >= self.max_focal_lane_hops:
                    continue
                for neighbor_id in neighbors[current_id]:
                    if neighbor_id not in connected_lane_ids:
                        connected_lane_ids.add(neighbor_id)
                        queue.append((neighbor_id, hops + 1))

        # Add the connected lane segments to the features under "focal_lane"
        self.features["focal_lane"].extend([
            CartesianPolyLine(
                locations=polyline.locations,  # This copies a mutable list
                location_type="focal_lane",
                name=polyline.name,
                identifier=polyline.identifier,
                covariance=polyline.covariance,
                origin=polyline.origin,
            )
            for polyline in lane_centerlines if int(polyline.name) in connected_lane_ids
        ])

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
        threshold_close: Distance in metres below which a moving agent is counted as "close" to the focal agent.
            All others are ignored for efficiency.
        threshold_topk: Maximum number of agents to include in features after distance filtering. If the number of
            close moving agents exceeds this limit, only the nearest ones (by min_dist_to_focal) are kept.
            Set to None to disable the cap.
    """

    def __init__(self,
        prediction_path: str | Path,
        split_path: str | Path,
        scenario_id: str,
        include_individual_agents: bool = True,
        threshold_is_moving: float = 0.25,
        threshold_close: float = 25.0,
        threshold_topk: int | None = 15,
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
        self.threshold_close = threshold_close
        self.threshold_topk = threshold_topk

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

        # focal rows are placed first by the pd.concat above, so fx/fy is always set before any non-focal agent is reached
        fx, fy = None, None
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

            is_focal = track_id == focal_track_id
            if is_focal:
                min_dist_to_focal = None
                fx, fy = np.array(x_traj), np.array(y_traj)
            elif fx is not None and len(x_traj) > 0:
                n = min(len(x_traj), len(fx))
                d = np.sqrt((np.array(x_traj[:n]) - fx[:n])**2 + (np.array(y_traj[:n]) - fy[:n])**2)
                min_dist_to_focal = float(d.min())
            else:
                print(f"Warning: Could not compute min_dist_to_focal for track_id {track_id} in scenario {self.scenario_id} (fx={fx is not None}, traj_len={len(x_traj)}).")
                min_dist_to_focal = 0.0

            state = {
                "track_id": track_id,
                "probability": prob,
                "x_traj": x_traj,
                "y_traj": y_traj,
                "agent_type": agent_type,
                "covariance": 1,
                "is_moving": is_moving,
                "min_dist_to_focal": min_dist_to_focal,
                "included_in_features": (
                    is_focal
                    or (
                        is_moving
                        and min_dist_to_focal is not None  # guaranteed to be True
                        and min_dist_to_focal <= self.threshold_close
                    )
                ),
            }

            if is_focal:
                self.focal_predictions.append(state)
            else:
                self.agents.append(state)

        if self.threshold_topk is not None:
            included = [ag for ag in self.agents if ag["included_in_features"]]
            if len(included) > self.threshold_topk:
                print(f"Number of included agents ({len(included)}) exceeds threshold_topk ({self.threshold_topk}). Applying top-k filtering based on min_dist_to_focal.")
                included.sort(key=lambda ag: ag["min_dist_to_focal"])
                for ag in included[self.threshold_topk:]:
                    ag["included_in_features"] = False

        # # Temporary debug: moving-agent count + min-distance distribution from focal agent
        # # (min distance per agent = closest approach; large min → safe to exclude)
        # moving_agents = [ag for ag in self.agents if ag["is_moving"]]
        # min_dists = [ag["min_dist_to_focal"] for ag in moving_agents]
        # if min_dists:
        #     s = pd.Series(min_dists, name="min_dist")
        #     stats = s.describe(percentiles=[0.25, 0.5, 0.75]).rename({"25%": "q25", "50%": "q50", "75%": "q75"})
        #     debug_row = stats[["count", "min", "q25", "q50", "mean", "q75", "max"]].to_frame().T
        #     debug_row["count"] = debug_row["count"].astype(int)
        #     n_included = sum(ag["included_in_features"] for ag in moving_agents)
        #     n_excluded = len(moving_agents) - n_included
        #     print(f"[moving agents] n={len(moving_agents)}/{len(self.agents)} | included={n_included} excluded={n_excluded} | " + " | ".join(f"{c}={debug_row[c].values[0]:.1f}" for c in debug_row.columns if c != "count"))
        # else:
        #     print(f"[moving agents] n=0/{len(self.agents)}")

        # self.agents = []

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

        standstill_agents = []
        for agent in self.agents:
            if not agent["included_in_features"]:
                continue

            x_traj = agent["x_traj"]
            y_traj = agent["y_traj"]

            index = min(self.current_index, len(x_traj)-1)

            x = x_traj[index].item()
            y = y_traj[index].item()
            # TODO: incorporate agent["probability"] for certainty in prediction (dropped in load() for simplicity for now)

            self.features[agent["agent_type"]].append(
                CartesianLocation(
                    east=x, north=y,
                    name=agent['track_id'],
                    location_type=agent["agent_type"],
                    covariance=agent["covariance"] * np.eye(2),
                )
            )
            if not agent["is_moving"]:
                standstill_agents.append((agent, index))
            else:
                if self.include_individual_agents:
                    self.features[f"agent_{agent['track_id']}"].append(
                        CartesianLocation(
                            east=x, north=y,
                            name=agent['track_id'],
                            location_type=f"agent_{agent['track_id']}",
                            covariance=agent["covariance"] * np.eye(2),
                        )
                    )

        # standstill_agents is already filtered for distance to focal agent
        self.features["standing_vehicle"].extend([
            CartesianLocation(
                east=agent["x_traj"][index].item(),
                north=agent["y_traj"][index].item(),
                name=agent['track_id'],
                location_type="standing_vehicle",
                covariance=agent["covariance"] * np.eye(2),
            )
            for agent, index in standstill_agents if agent["agent_type"].lower() in ["vehicle", "bus"]
        ])

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
                if not agent["included_in_features"]:
                    continue

                agent_name = f"agent_{agent['track_id']}"

                # Require "distance(agent_123) <- source("/star_map/distance/agent_123", Density)."
                facts += f"distance({agent_name}) <- source(\"/star_map/distance/{agent_name}\", Density).\n"

                # Facts like "vehicle(agent_123).", "pedestrian(agent_456).", etc.
                facts += f"{agent['agent_type']}({agent_name}).\n"

        return facts

    @cached_property
    def focal_starting_location(self) -> CartesianLocation:
        """Return the starting location of the focal agent as a CartesianLocation."""
        # We just pick one of the predictions for the starting location, since they all start at the same point
        state = self.focal_predictions[0]
        x = state["x_traj"][0].item()
        y = state["y_traj"][0].item()
        return CartesianLocation(east=x, north=y, name=self.scenario_id, location_type="focal_agent_start")
