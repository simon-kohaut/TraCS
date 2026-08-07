"""Agentic LLM pipeline for generating Resin compliance programs from natural-language regulations."""

#
# Copyright (c) Simon Kohaut, Felix Divo, Julius Hahnewald, and contributors
#
# This file is part of TraCS and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with TraCS.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

from dataclasses import dataclass
from os import getenv
from time import perf_counter
from typing import Callable, Optional

from openai import LengthFinishReasonError

from langchain.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from tracs.agents.resin_verifier import ResinSyntaxChecker


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
    coder_seconds: float
    judge_seconds: float
    verifier_seconds: float
    seconds: float
    coder_usage: dict
    judge_usage: dict


class CoderPipeline:
    """Three-agent iterative pipeline: coder -> judge + compiler -> (repeat until accepted)."""

    def __init__(
        self,
        *,
        coder_backend: str,
        coder_model: str,
        judge_backend: str,
        judge_model: str,
        max_iterations: int = 10,
        coder_temperature: float = 0.3,
        judge_temperature: float = 0.0,
        seed: Optional[int] = None,
        max_tokens: int = 20_000,
        history_window: int = 3,
    ):
        self.coder_backend = coder_backend
        self.coder_model = coder_model
        self.judge_backend = judge_backend
        self.judge_model = judge_model
        self.max_iterations = max_iterations
        self.coder_temperature = coder_temperature
        self.judge_temperature = judge_temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.history_window = history_window

        self._coder = self._build_model(
            coder_backend, coder_model, coder_temperature, seed, max_tokens=max_tokens
        )
        self._judge = self._build_model(
            judge_backend, judge_model, judge_temperature, seed, max_tokens=max_tokens
        ).with_structured_output(JudgeOutput, include_raw=True)
        self._verifier = ResinSyntaxChecker()

        with open("prompt/system_coder.txt") as f:
            self._coder_system = f.read()
        with open("prompt/system_judge.txt") as f:
            self._judge_system = f.read()

    def _build_model(
        self,
        backend: str,
        model: str,
        temperature: float,
        seed: Optional[int],
        *,
        max_tokens: int,
    ) -> BaseChatModel:
        match backend:
            case "openai":
                return ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,
                )
            case "anthropic":
                # Some Anthropic models reject the `temperature` parameter, so we omit it.
                return ChatAnthropic(model=model, max_tokens=max_tokens)
            case "google":
                return ChatGoogleGenerativeAI(
                    model=model,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,
                )
            case "local":
                return ChatOpenAI(
                    model=model,
                    base_url=getenv(
                        "LOCAL_MODEL_BASE_URL", "http://host.docker.internal:54112/v1"
                    ),
                    api_key=getenv("LOCAL_MODEL_KEY", ""),
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,
                )
            case "ollama":
                return ChatOllama(
                    model=model,
                    temperature=temperature,
                    seed=seed,
                    num_predict=max_tokens,
                )
            case _:
                raise ValueError(f"Unknown backend: {backend!r}")

    def run(
        self,
        regulation: str,
        on_iteration: Optional[Callable[[IterationRecord], None]] = None,
    ) -> tuple[str, list[IterationRecord]]:
        """Run the pipeline to produce an accepted Resin program.

        The coder retains conversation history across iterations, so it can
        see all of its previous attempts and the critiques they received.
        The judge stays stateless (re-prompted from scratch each iteration).

        on_iteration is called after each iteration with the completed record,
        before deciding whether to continue. Returns the accepted program and
        the full iteration log. Raises RuntimeError if max_iterations is reached.
        """
        coder_messages: list = [
            SystemMessage(content=self._coder_system),
            HumanMessage(content=self._initial_coder_prompt(regulation)),
        ]
        history: list[IterationRecord] = []

        for i in range(self.max_iterations):
            iteration_start = perf_counter()

            program, coder_usage, coder_seconds = self._invoke_coder(coder_messages)
            coder_messages.append(AIMessage(content=program))

            judge_passed, critique, judge_usage, judge_seconds = self._call_judge(regulation, program)

            verifier_start = perf_counter()
            compiled, diagnostics = self._verifier.verify(program)
            verifier_seconds = perf_counter() - verifier_start

            record = IterationRecord(
                iteration=i + 1,
                program=program,
                judge_passed=judge_passed,
                critique=critique,
                compiled=compiled,
                diagnostics=diagnostics,
                coder_seconds=coder_seconds,
                judge_seconds=judge_seconds,
                verifier_seconds=verifier_seconds,
                seconds=perf_counter() - iteration_start,
                coder_usage=coder_usage,
                judge_usage=judge_usage,
            )
            history.append(record)

            if on_iteration:
                on_iteration(record)

            if judge_passed and compiled:
                return program, history

            feedback = self._build_feedback(
                judge_passed, critique, compiled, diagnostics
            )
            coder_messages.append(HumanMessage(content=self._revision_prompt(feedback)))

        raise RuntimeError(
            f"Pipeline did not converge after {self.max_iterations} iterations."
        )

    def _invoke_coder(self, messages: list) -> tuple[str, dict, float]:
        start = perf_counter()
        message = self._coder.invoke(self._trim_history(messages))
        seconds = perf_counter() - start
        program = _strip_code_fences(_strip_thinking_traces(_content_to_str(message.content).strip()))
        return program, _usage_to_dict(message), seconds

    def _trim_history(self, messages: list) -> list:
        """Keep system + initial prompt and the last history_window revision cycles."""
        header = messages[:2]
        tail = messages[2:]
        max_tail = self.history_window * 2
        if len(tail) > max_tail:
            print(f"WARNING: trimming coder history to last {self.history_window} cycles")
            tail = tail[-max_tail:]
        return header + tail

    def _call_judge(self, regulation: str, program: str) -> tuple[bool, str, dict, float]:
        messages = [
            SystemMessage(content=self._judge_system),
            HumanMessage(content=self._judge_prompt(regulation, program)),
        ]
        start = perf_counter()
        try:
            result = self._judge.invoke(messages)
        except LengthFinishReasonError:
            seconds = perf_counter() - start
            return False, "The judge did not conclude within the token budget. Possibly, your program was too long.", {}, seconds
        seconds = perf_counter() - start

        parsed: Optional[JudgeOutput] = result["parsed"]
        if parsed is None:
            raise result["parsing_error"]
        return parsed.verdict, parsed.critique, _usage_to_dict(result["raw"]), seconds

    @staticmethod
    def _initial_coder_prompt(regulation: str) -> str:
        return f"Traffic regulation (natural language):\n{regulation}"

    @staticmethod
    def _revision_prompt(feedback: str) -> str:
        return (
            "Your previous attempt was rejected. "
            f"Revise your program based on this feedback:\n{feedback}"
        )

    @staticmethod
    def _judge_prompt(regulation: str, program: str) -> str:
        return (
            f"Traffic regulation (natural language):\n{regulation}"
            "\n\n"
            f"Candidate Resin program:\n{program}"
        )

    @staticmethod
    def _build_feedback(
        judge_passed: bool, critique: str, compiled: bool, diagnostics: str
    ) -> str:
        parts = []
        if not judge_passed:
            parts.append(f"Alignment critique:\n{critique}")
        if not compiled:
            parts.append(f"Compiler errors:\n{diagnostics}")
        return "\n\n".join(parts)


def _usage_to_dict(message) -> dict:
    """Token counts from a LangChain response; None where the backend reports none."""
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


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


def _strip_thinking_traces(text: str) -> str:
    """Remove chain-of-thought traces that some models (e.g. Qwen3.5) emit."""
    if "</think>" not in text:
        return text
    print("WARNING: coder returned thinking traces — stripping")
    return text.split("</think>", 1)[-1].strip()


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
