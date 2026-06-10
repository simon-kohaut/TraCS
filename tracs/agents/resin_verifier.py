"""Resin language syntax checker.

Pure parsing/validation — no LLM dependencies. Used by the pipeline to give
the coder structured feedback when its output fails to compile.
"""

#
# Copyright (c) Simon Kohaut, Felix Divo, Julius Hahnewald, and contributors
#
# This file is part of TraCS and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with TraCS.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

import re


class ResinSyntaxChecker:
    """Syntax checker for the Resin language."""

    _IDENTIFIER = r"[a-zA-Z_][a-zA-Z0-9_]*"
    _NUMBER = r"[+-]?\d+(?:\.\d*)?"
    _ARG_LIST = rf"(?:{_IDENTIFIER}|{_NUMBER})(?:\s*,\s*(?:{_IDENTIFIER}|{_NUMBER}))*"
    _PREDICATE = rf"{_IDENTIFIER}(?:\s*\(\s*(?:{_ARG_LIST})?\s*\))?"
    _COMPARISON_OP = r"(?:[<>]=?|==)"
    _CONDITION = (
        rf"(?:not\s+)?(?:{_PREDICATE})(?:\s*{_COMPARISON_OP}\s*{_NUMBER})?"
        rf"|{_NUMBER}\s*{_COMPARISON_OP}\s*(?:{_PREDICATE})"
    )

    _HEAD_RE = re.compile(rf"^\s*{_PREDICATE}\s*$")
    _CONDITION_RE = re.compile(rf"^\s*{_CONDITION}\s*$")

    def verify(self, program: str) -> tuple[bool, str]:
        """Check Resin program syntax. Returns (success, diagnostics)."""
        errors = []

        # Strip inline comments
        clean_lines = []
        for line in program.splitlines():
            idx = line.find("#")
            clean_lines.append(line[:idx] if idx != -1 else line)
        clean_text = "\n".join(clean_lines)

        clauses = self._split_clauses(clean_text)

        has_landscape = False
        clause_count = 0

        for clause_text, line_num in clauses:
            clause_text = clause_text.strip()
            if not clause_text:
                continue
            clause_count += 1

            if self._clause_head(clause_text) == "landscape":
                has_landscape = True

            for err in self._check_clause(clause_text):
                errors.append(f"Line {line_num}: {err}")

        if clause_count == 0:
            errors.append("Program contains no clauses.")

        if not has_landscape:
            errors.append("Missing required 'landscape' definition.")

        return (not bool(errors), "\n".join(errors))

    @staticmethod
    def _split_clauses(text: str) -> list[tuple[str, int]]:
        """Split Resin source into (clause_text, start_line) pairs.

        Clause terminators are '.' not followed by a digit, so decimal points
        inside numbers (digit.digit) are never treated as terminators.
        """
        clauses = []
        prev_end = 0
        for m in re.finditer(r"\.(?!\d)", text):
            clause = text[prev_end: m.start()]
            line_num = text[:prev_end].count("\n") + 1
            clauses.append((clause, line_num))
            prev_end = m.end()
        remaining = text[prev_end:].strip()
        if remaining:
            clauses.append((remaining, text[:prev_end].count("\n") + 1))
        return clauses

    @staticmethod
    def _clause_head(clause: str) -> str:
        """Return the head predicate name of a clause."""
        head_part = re.split(r"\bif\b", clause, maxsplit=1)[0].strip()
        m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)", head_part)
        return m.group(1) if m else ""

    @classmethod
    def _check_clause(cls, clause: str) -> list[str]:
        """Validate one clause; return a list of error strings."""
        errors = []
        parts = re.split(r"\bif\b", clause, maxsplit=1)
        head = parts[0].strip()
        body = parts[1].strip() if len(parts) == 2 else None

        if not cls._HEAD_RE.match(head):
            errors.append(f"Invalid head: '{head}'")

        if body is not None:
            for cond in re.split(r"\band\b", body):
                cond = cond.strip()
                if cond and not cls._CONDITION_RE.match(cond):
                    errors.append(f"Invalid condition: '{cond}'")

        return errors
