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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


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


def compile_resin(program: str) -> tuple[bool, str]:
    """Check Resin program syntax. Returns (success, diagnostics)."""
    errors = []

    # Strip inline comments
    clean_lines = []
    for line in program.splitlines():
        idx = line.find("#")
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
    for m in re.finditer(r"\.(?!\d)", text):
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
    head_part = re.split(r"\bif\b", clause, maxsplit=1)[0].strip()
    m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)", head_part)
    return m.group(1) if m else ""


def _check_clause(clause: str) -> list[str]:
    """Validate one clause; return a list of error strings."""
    errors = []
    parts = re.split(r"\bif\b", clause, maxsplit=1)
    head = parts[0].strip()
    body = parts[1].strip() if len(parts) == 2 else None

    if not _HEAD_RE.match(head):
        errors.append(f"Invalid head: '{head}'")

    if body is not None:
        for cond in re.split(r"\band\b", body):
            cond = cond.strip()
            if cond and not _CONDITION_RE.match(cond):
                errors.append(f"Invalid condition: '{cond}'")

    return errors


class JudgeOutput(BaseModel):
    verdict: bool
    critique: str


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
    coder_temperature: float = 0.3
    judge_temperature: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self):
        self._coder = self._build_model(
            self.coder_model, self.coder_temperature, self.seed
        )
        self._judge = self._build_model(
            self.judge_model, self.judge_temperature, self.seed
        ).with_structured_output(JudgeOutput)

    def _build_model(self, model: str, temperature: float, seed: Optional[int]):
        match self.backend:
            case "openai":
                return ChatOpenAI(model=model, temperature=temperature, seed=seed)
            case "anthropic":
                return ChatAnthropic(
                    model=model, temperature=temperature, max_tokens=4096
                )
            case "google":
                return ChatGoogleGenerativeAI(model=model, temperature=temperature)
            case "local":
                return ChatOpenAI(
                    model=model,
                    base_url=getenv(
                        "LOCAL_MODEL_BASE_URL", "http://host.docker.internal:54112/v1"
                    ),
                    api_key=getenv("LOCAL_MODEL_KEY", ""),
                    temperature=temperature,
                    seed=seed,
                )
            case "ollama":
                return ChatOllama(model=model, temperature=temperature, seed=seed)
            case _:
                raise ValueError(f"Unknown backend: {self.backend!r}")

    def run(
        self,
        regulation: str,
        on_iteration: Optional[Callable[["IterationRecord"], None]] = None,
    ) -> tuple[str, list["IterationRecord"]]:
        """Run the pipeline to produce an accepted Resin program.

        on_iteration is called after each iteration with the completed record,
        before deciding whether to continue. Returns the accepted program and
        the full iteration log. Raises RuntimeError if max_iterations is reached.
        """
        history: list[IterationRecord] = []
        feedback: Optional[tuple[str, str]] = None

        for i in range(self.max_iterations):
            program = self._call_coder(regulation, feedback)
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

            feedback = _build_feedback(
                judge_passed, critique, compiled, diagnostics, program
            )

        raise RuntimeError(
            f"Pipeline did not converge after {self.max_iterations} iterations."
        )

    def _call_coder(self, regulation: str, feedback: Optional[str]) -> str:
        with open("prompt/system_coder.txt") as f:
            _CODER_SYSTEM = f.read()
        messages = [
            SystemMessage(content=_CODER_SYSTEM),
            HumanMessage(content=_coder_prompt(regulation, feedback)),
        ]
        content = self._coder.invoke(messages).content
        return _strip_code_fences(_content_to_str(content).strip())

    def _call_judge(self, regulation: str, program: str) -> tuple[bool, str]:
        with open("prompt/system_judge.txt") as f:
            _JUDGE_SYSTEM = f.read()
        messages = [
            SystemMessage(content=_JUDGE_SYSTEM),
            HumanMessage(content=_judge_prompt(regulation, program)),
        ]
        result: JudgeOutput = self._judge.invoke(messages)
        return result.verdict, result.critique


def _coder_prompt(regulation: str, feedback: Optional[str]) -> str:
    prompt = f"Traffic regulation (natural language):\n{regulation}"
    if feedback:
        prompt += (
            "\n\nYour previous attempt was rejected. "
            f"Revise your program based on this feedback:\n{feedback}"
        )
    return prompt


def _judge_prompt(regulation: str, program: str) -> str:
    return (
        f"Traffic regulation (natural language):\n{regulation}"
        "\n\n"
        f"Candidate Resin program:\n{program}"
    )


def _build_feedback(
    judge_passed: bool, critique: str, compiled: bool, diagnostics: str, program: str
) -> tuple[str, str]:
    parts = []
    if not judge_passed:
        parts.append(f"Alignment critique:\n{critique}")
    if not compiled:
        parts.append(f"Compiler errors:\n{diagnostics}")
    if parts:
        parts.append(f"Faulty program:\n{program}")
    return "\n\n".join(parts)


def _content_to_str(content) -> str:
    """Normalize a LangChain message content to a flat string.

    OpenAI/Anthropic return a string; Gemini returns a list of content parts.
    """
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            parts.append(part.get("text", ""))
    return "".join(parts)


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes add despite instructions."""
    if "```" not in text:
        return text
    print("WARNING: coder returned markdown fences — stripping")
    blocks = text.split("```")
    for idx, block in enumerate(blocks):
        if idx % 2 == 1:
            lines = block.splitlines()
            # Drop the language tag line (e.g. "resin") if present
            if lines and not lines[0].strip().startswith(("%", ":-", "?")):
                lines = lines[1:]
            return "\n".join(lines).strip()
    return text
