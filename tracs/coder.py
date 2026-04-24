"""Agentic LLM pipeline for generating Resin compliance programs from natural-language regulations."""

#
# Copyright (c) Simon Kohaut, Felix Divo, Julius Hahnewald, and contributors
#
# This file is part of TraCS and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with TraCS.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

import re
from dataclasses import dataclass
from os import getenv
from typing import Callable, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


_CODER_SYSTEM = """\
You are an expert in probabilistic logic programming using the Resin language.
Given a natural-language description of traffic regulations and a library of
probabilistic spatial relations, generate a Resin program that formally encodes
the described rules using only the primitives provided by that library.

Here is an example Resin program:
    permitted if over(park).
    permitted if over(water).

    road_safety if distance(primary) < 35.0.
    road_safety if distance(secondary) < 15.0.

    agent_safety if distance(vessel) > 100.0 and distance(uas) > 100.0.

    landscape if distance(hospital) > 200.0 and road_safety and agent_safety.
    landscape if permitted and agent_safety.

Output ONLY the raw Resin program. No prose, no markdown fences.
"""

_JUDGE_SYSTEM = """\
You are an expert in traffic regulations and probabilistic logic programming.
Given a natural-language regulation and a candidate Resin program, evaluate
whether the program faithfully encodes the regulation. Identify logical gaps,
missing constraints, or misrepresentations.

The regulations aim to express rules for a single agent relative to it and its environment.
Do not expect rules to reason about an entire traffic space and all combinations of agents.
All relations such as distnace(T) imply that they hold for ALL environment features of that type T.

Respond in exactly this format:
VERDICT: PASS
CRITIQUE: OK

or

VERDICT: FAIL
CRITIQUE: <concise explanation of what is wrong or missing>
"""


_IDENTIFIER    = r'[a-zA-Z_][a-zA-Z0-9_]*'
_NUMBER        = r'[+-]?\d+(?:\.\d*)?'
_ARG_LIST      = rf'(?:{_IDENTIFIER}|{_NUMBER})(?:\s*,\s*(?:{_IDENTIFIER}|{_NUMBER}))*'
_PREDICATE     = rf'{_IDENTIFIER}(?:\s*\(\s*(?:{_ARG_LIST})?\s*\))?'
_COMPARISON_OP = r'(?:[<>]=?|[!=]=)'
_CONDITION     = rf'(?:{_PREDICATE})(?:\s*{_COMPARISON_OP}\s*{_NUMBER})?'

_HEAD_RE      = re.compile(rf'^\s*{_PREDICATE}\s*$')
_CONDITION_RE = re.compile(rf'^\s*{_CONDITION}\s*$')


def compile_resin(program: str) -> tuple[bool, str]:
    """Check Resin program syntax. Returns (success, diagnostics)."""
    errors = []

    # Strip inline comments
    clean_lines = []
    for line in program.splitlines():
        idx = line.find('#')
        clean_lines.append(line[:idx] if idx != -1 else line)
    clean_text = "\n".join(clean_lines)

    clauses = _split_resin_clauses(clean_text)

    has_landscape = False
    clause_count = 0

    for clause_text, line_num in clauses:
        clause_text = clause_text.strip()
        if not clause_text:
            continue
        clause_count += 1

        if _clause_head(clause_text) == "landscape":
            has_landscape = True

        for err in _check_clause(clause_text):
            errors.append(f"Line {line_num}: {err}")

    if clause_count == 0:
        errors.append("Program contains no clauses.")

    if not has_landscape:
        errors.append("Missing required 'landscape' definition.")

    return (not bool(errors), "\n".join(errors))


def _split_resin_clauses(text: str) -> list[tuple[str, int]]:
    """Split Resin source into (clause_text, start_line) pairs.

    Clause terminators are '.' not followed by a digit, so decimal points
    inside numbers (digit.digit) are never treated as terminators.
    """
    clauses = []
    prev_end = 0
    for m in re.finditer(r'\.(?!\d)', text):
        clause = text[prev_end : m.start()]
        line_num = text[:prev_end].count("\n") + 1
        clauses.append((clause, line_num))
        prev_end = m.end()
    remaining = text[prev_end:].strip()
    if remaining:
        clauses.append((remaining, text[:prev_end].count("\n") + 1))
    return clauses


def _clause_head(clause: str) -> str:
    """Return the head predicate name of a clause."""
    head_part = re.split(r'\bif\b', clause, maxsplit=1)[0].strip()
    m = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', head_part)
    return m.group(1) if m else ""


def _check_clause(clause: str) -> list[str]:
    """Validate one clause; return a list of error strings."""
    errors = []
    parts = re.split(r'\bif\b', clause, maxsplit=1)
    head = parts[0].strip()
    body = parts[1].strip() if len(parts) == 2 else None

    if not _HEAD_RE.match(head):
        errors.append(f"Invalid head: '{head}'")

    if body is not None:
        for cond in re.split(r'\band\b', body):
            cond = cond.strip()
            if cond and not _CONDITION_RE.match(cond):
                errors.append(f"Invalid condition: '{cond}'")

    return errors


@dataclass
class IterationRecord:
    iteration: int
    program: str
    judge_passed: bool
    critique: str
    compiled: bool
    diagnostics: str


@dataclass
class CoderPipeline:
    """Three-agent iterative pipeline: coder -> judge -> compiler -> (repeat until accepted)."""

    coder_model: str = "qwen2.5:7b-instruct"
    judge_model: str = "qwen2.5:7b-instruct"
    backend: str = "ollama"
    max_iterations: int = 10
    coder_temperature: float = 0.2
    judge_temperature: float = 0.0

    def __post_init__(self):
        self._coder = self._build_model(self.coder_model, self.coder_temperature)
        self._judge = self._build_model(self.judge_model, self.judge_temperature)

    def _build_model(self, model: str, temperature: float):
        if self.backend == "openai":
            return ChatOpenAI(model=model, temperature=temperature)
        if self.backend == "anthropic":
            return ChatAnthropic(model=model, temperature=temperature, max_tokens=4096)
        if self.backend == "local":
            return ChatOpenAI(
                model=model,
                base_url=getenv("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:54112/v1"),
                api_key=getenv("LOCAL_MODEL_KEY", ""),
                temperature=temperature,
            )
        return ChatOllama(model=model, temperature=temperature)

    def run(
        self,
        regulation: str,
        library: str,
        on_iteration: Optional[Callable[["IterationRecord"], None]] = None,
    ) -> tuple[str, list["IterationRecord"]]:
        """Run the pipeline to produce an accepted Resin program.

        on_iteration is called after each iteration with the completed record,
        before deciding whether to continue. Returns the accepted program and
        the full iteration log. Raises RuntimeError if max_iterations is reached.
        """
        history: list[IterationRecord] = []
        feedback: Optional[str] = None

        for i in range(self.max_iterations):
            program = self._call_coder(regulation, library, feedback)
            judge_passed, critique = self._call_judge(regulation, program)
            compiled, diagnostics = compile_resin(program)

            record = IterationRecord(
                iteration=i + 1,
                program=program,
                judge_passed=judge_passed,
                critique=critique,
                compiled=compiled,
                diagnostics=diagnostics,
            )
            history.append(record)

            if on_iteration:
                on_iteration(record)

            if judge_passed and compiled:
                return program, history

            feedback = _build_feedback(judge_passed, critique, compiled, diagnostics)

        raise RuntimeError(
            f"Pipeline did not converge after {self.max_iterations} iterations."
        )

    def _call_coder(self, regulation: str, library: str, feedback: Optional[str]) -> str:
        messages = [
            SystemMessage(content=_CODER_SYSTEM),
            HumanMessage(content=_coder_prompt(regulation, library, feedback)),
        ]
        response = self._coder.invoke(messages).content.strip()
        return _strip_code_fences(response)

    def _call_judge(self, regulation: str, program: str) -> tuple[bool, str]:
        messages = [
            SystemMessage(content=_JUDGE_SYSTEM),
            HumanMessage(content=_judge_prompt(regulation, program)),
        ]
        response = self._judge.invoke(messages).content.strip()
        return _parse_judge_response(response)


def _coder_prompt(regulation: str, library: str, feedback: Optional[str]) -> str:
    prompt = (
        f"Spatial relation library:\n{library}\n\n"
        f"Traffic regulation:\n{regulation}\n\n"
        "Generate a Resin program that encodes this regulation using the library above."
    )
    if feedback:
        prompt += (
            "\n\nYour previous attempt was rejected. "
            f"Revise your program based on this feedback:\n{feedback}"
        )
    return prompt


def _judge_prompt(regulation: str, program: str) -> str:
    return (
        f"Traffic regulation (natural language):\n{regulation}\n\n"
        f"Candidate Resin program:\n{program}\n\n"
        "Does this program faithfully encode the regulation?"
    )


def _build_feedback(
    judge_passed: bool, critique: str, compiled: bool, diagnostics: str
) -> str:
    parts = []
    if not judge_passed:
        parts.append(f"Alignment critique:\n{critique}")
    if not compiled:
        parts.append(f"Compiler errors:\n{diagnostics}")
    return "\n\n".join(parts)


def _parse_judge_response(response: str) -> tuple[bool, str]:
    """Parse VERDICT/CRITIQUE lines from the judge's response."""
    verdict = False
    critique = response
    for line in response.splitlines():
        upper = line.upper()
        if upper.startswith("VERDICT:"):
            verdict = "PASS" in upper
        elif upper.startswith("CRITIQUE:"):
            critique = line.split(":", 1)[1].strip()
    return verdict, critique


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes add despite instructions."""
    if "```" not in text:
        return text
    blocks = text.split("```")
    for idx, block in enumerate(blocks):
        if idx % 2 == 1:
            lines = block.splitlines()
            # Drop the language tag line (e.g. "prolog") if present
            if lines and not lines[0].strip().startswith(("%", ":-", "?")):
                lines = lines[1:]
            return "\n".join(lines).strip()
    return text
