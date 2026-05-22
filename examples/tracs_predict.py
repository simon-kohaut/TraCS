import multiprocessing
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

from rich.progress import MofNCompleteColumn, Progress, ProgressColumn
from rich.text import Text

from eval import tracs_rate_scenario


class IterationsPerSecondColumn(ProgressColumn):
    def render(self, task) -> Text:
        speed = task.speed
        if speed is None:
            return Text("? it/s", style="progress.data.speed")
        return Text(f"{speed:.2f} it/s", style="progress.data.speed")


tracs_logic = """
distance(vehicle) <- source("/star_map/distance/vehicle", Density).
speed <- source("/sensors/speed", Number).
acceleration <- source("/sensors/acceleration", Number).
distance(standing_vehicle) <- source("/star_map/distance/standing_vehicle", Density).
follows(focal_lane) <- source("/star_map/follows/focal_lane", Probability).

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

safe if vehicle(V) and distance(V) > 10.
landscape if distance(standing_vehicle) > 2 and speed < 50 and safe and follows(focal_lane).

landscape -> target("/landscape").
"""


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


def process_scenario(
    scenario_id: str,
    logic: str,
    prediction_path: Path,
    data_root: Path,
    split: str,
    output_dir: Path,
) -> None:
    outout_file = output_dir / f"{scenario_id}.parquet"
    if not outout_file.exists():
        scored_predictions, _ = tracs_rate_scenario(
            scenario_id,
            logic=logic,
            prediction_path=prediction_path,
            data_root=data_root,
            split=split,
        )
        scored_predictions.to_parquet(outout_file)


def main(args) -> None:
    submission_file = args.data_root / "av2_backbone_outputs" / args.backbone / "submission.parquet"
    output_dir = args.output_dir / args.split / args.backbone
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = dict(logic=tracs_logic, prediction_path=submission_file,
                  data_root=args.data_root, split=args.split, output_dir=output_dir)

    scenario_ids = [file.stem for file in (args.data_root / "av2" / args.split).iterdir()]

    try:
        with Progress(*Progress.get_default_columns(), MofNCompleteColumn(), IterationsPerSecondColumn()) as progress:
            task = progress.add_task("Processing scenarios...", total=len(scenario_ids))
            if args.workers is not None and args.workers > 1:
                with multiprocessing.Pool(args.workers) as p:
                    for _ in p.imap_unordered(partial(process_scenario, **kwargs), scenario_ids):
                        progress.advance(task)
            else:
                for sid in scenario_ids:
                    process_scenario(sid, **kwargs)
                    progress.advance(task)

    except KeyboardInterrupt:
        print("Process interrupted by user. Exiting...")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--backbone", type=str, default="DeMo")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output_dir", type=Path, default=Path("data") / "tracs_predictions" / "simple-logic")
    # TODO consume JSON config file for logic
    args = parser.parse_args()

    main(args)
