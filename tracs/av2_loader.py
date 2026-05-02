"""Loaders for fetching spatial data from static AV2 Maps."""

#
# Copyright (c) Simon Kohaut, Felix Divo, Julius Hahnewald, and contributors
#
# This file is part of TraCS and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with TraCS.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Standard Library
import logging
from pathlib import Path

# Third Party
from av2.map.map_api import ArgoverseStaticMap
from shapely import total_bounds

# ProMis
from promis.geo import CartesianLocation, CartesianPolygon, CartesianPolyLine, CartesianMap
from promis.loaders.spatial_loader import SpatialLoader

log = logging.getLogger(__name__)

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
        self.features = []

        # Load the Argoverse 2 map API
        self._av2_map = ArgoverseStaticMap.from_json(self.path)
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
        for da in self._av2_map.vector_drivable_areas.values():
            coords = [CartesianLocation(east=pt[0], north=pt[1]) for pt in da.xyz]
            self.features.append(CartesianPolygon(coords, location_type=location_type))

    def _load_pedestrian_crossings(self, inverted_fd: dict[str, str]) -> None:
        location_type = inverted_fd["PEDESTRIAN_CROSSING"]
        for pc in self._av2_map.vector_pedestrian_crossings.values():
            coords = [CartesianLocation(east=pt[0], north=pt[1]) for pt in pc.polygon]
            self.features.append(CartesianPolygon(coords, location_type=location_type))

    def _load_lane_segments(self, inverted_fd: dict[str, str]) -> None:
        for ls in self._av2_map.vector_lane_segments.values():
            lane_type = inverted_fd.get(ls.lane_type.value, str(ls.lane_type.value))

            coords = [CartesianLocation(east=pt[0], north=pt[1]) for pt in ls.polygon_boundary]
            self.features.append(CartesianPolygon(coords, location_type=lane_type))
            if ls.is_intersection:
                self.features.append(CartesianPolygon(coords, location_type=inverted_fd.get("INTERSECTION", "INTERSECTION")))

            centerline = self._av2_map.get_lane_segment_centerline(ls.id)
            center_coords = [CartesianLocation(east=pt[0], north=pt[1]) for pt in centerline]
            self.features.append(CartesianPolyLine(center_coords, location_type=f"{lane_type}_center"))

            for lane_marking in [ls.left_lane_marking, ls.right_lane_marking]:
                lane_marking_type = inverted_fd.get(lane_marking.mark_type.value, str(lane_marking.mark_type.value))
                coords = [CartesianLocation(east=pt[0], north=pt[1]) for pt in lane_marking.polyline]
                self.features.append(CartesianPolyLine(coords, location_type=lane_marking_type))

    def to_cartesian_map(self) -> CartesianMap:
        """Creates a CartesianMap from the loaded features.

        Returns:
            A CartesianMap containing all loaded features, with no known origin.
        """
        return CartesianMap(None, self.features)
