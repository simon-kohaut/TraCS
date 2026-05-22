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

safe if vehicle(V) and distance(V) > 10.
landscape if safe and speed < 50.

landscape -> target("/landscape").
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
