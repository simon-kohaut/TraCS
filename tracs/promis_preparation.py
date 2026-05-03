"""Generate full resin programs with headers and footers."""

import re


_RELATIONS = {
    "over": "Probability",
    "distance": "Density",
    "intersects": "Probability",
    "enters": "Probability",
    "follows": "Probability",
    "opposes": "Probability",
}


def _referenced_relations(rules: str) -> set[tuple[str, str]]:
    """Return the set of (relation, feature) pairs referenced in the rules."""
    pattern = re.compile(
        r"\b(" + "|".join(_RELATIONS) + r")\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)"
    )
    return set(pattern.findall(rules))


def _build_source_preamble(rules: str) -> str:
    """Emit one source declaration per (relation, feature) pair used in the rules."""
    lines = [
        f'{relation}({feature}) <- source("/star_map/{relation}/{feature}", {_RELATIONS[relation]}).'
        for relation, feature in sorted(_referenced_relations(rules))
    ]
    return "\n".join(lines)


def generate_full_program(rules: str) -> str:
    """Wrap a ProMis rule body into a complete, executable program.

    Scans ``rules`` for relation/feature references, auto-generates the
    matching ``source(...)`` declarations as a preamble, then appends a
    ``landscape -> target(...)`` footer so the program can be run directly
    by the ProMis engine.

    Args:
        rules: One or more ProMis clauses that define the landscape predicate.

    Returns:
        A self-contained ProMis program string (preamble + rules + footer).
    """

    return "\n".join(
        [
            _build_source_preamble(rules),
            "",  # Newline
            rules,
            "",  # Newline
            'landscape -> target("/landscape").',
        ]
    )
