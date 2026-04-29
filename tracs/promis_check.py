"""Prototype: verify Resin rules by evaluating them on a 1x1 ProMiS grid."""

from math import prod
import re

from promis import ProMis, StaRMap
from promis.geo import CartesianMap, CartesianRasterBand, PolarLocation


_RELATIONS = {
    "over":       "Probability",
    "distance":   "Density",
    "intersects": "Probability",
    "follows":    "Probability",
    "opposes":    "Probability",
}


def _referenced_relations(rules: str) -> set[tuple[str, str]]:
    """Return the set of (relation, feature) pairs referenced in the rules."""
    pattern = re.compile(
        r"\b(" + "|".join(_RELATIONS) +
        r")\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)"
    )
    return set(pattern.findall(rules))


def _build_source_preamble(rules: str) -> str:
    """Emit one source declaration per (relation, feature) pair used in the rules."""
    lines = [
        f'{relation}({feature}) <- source("/star_map/{relation}/{feature}", {_RELATIONS[relation]}).'
        for relation, feature in sorted(_referenced_relations(rules))
    ]
    return "\n".join(lines)


def check_with_promis(rules: str) -> tuple[bool, str]:
    """Wrap rules with source/target boilerplate and run them through ProMiS.

    Returns (success, message). On success the message is the probability
    at the single grid point; on failure it is the exception class + text.
    """
    program = "\n".join([
        _build_source_preamble(rules),
        rules,
        'landscape -> target("/landscape").',
    ])
    # print("=== ProMiS program ===")
    # print(program)
    # print("=====================")

    origin = PolarLocation(latitude=0.0, longitude=0.0)
    # Qhull needs at least 4 points for Delaunay triangulation, so use a 2x2 grid.
    grid = (2, 2)
    eval_points = CartesianRasterBand(origin, grid, 1.0, 1.0)
    star_map = StaRMap(CartesianMap(origin))

    try:
        star_map.initialize(
            eval_points, number_of_random_maps=1, logic=program)
        promis = ProMis(star_map, logic=program, dimension=prod(grid))
        promis.initialize(eval_points)
        result = promis.update()

        if result is None:
            return False, "ProMiS returned no result."

        prob = float(result.data["v0"].iloc[0])
        return True, f"probability={prob:.6f}"

    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


if __name__ == "__main__":
    print("valid:  ", check_with_promis("landscape if over(primary)."))
    print("garbage:", check_with_promis("@@@ this is not resin!? @@@"))
    print("missing:", check_with_promis("landscape if over_under(primary)"))
    print("missing:", check_with_promis("landscape if over(thing_which_is_no_there)"))
