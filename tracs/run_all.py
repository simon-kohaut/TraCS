"""Entry point for running all scenarios across all configured models and seeds."""

import subprocess
from pathlib import Path

OUT = Path("output")

# (backend, model)
MODELS = [
    ("local",      "google/gemma-4-31B-it"),
    # ("local",      "Qwen/Qwen3.6-27B"),
    # ("local",      "Aleph-Alpha/Pharia-1-LLM-7B-control-hf"),
    ("openai",     "gpt-5.5"),
    # ("openai",     "gpt-5.4-mini"),
    ("anthropic",  "claude-opus-4-7"),
    # ("anthropic",  "claude-sonnet-4-6"),
    # ("google",     "gemini-3.1-pro-preview"),
    ("google",     "gemini-3-flash-preview"),
]

SEEDS = [1] # , 2, 3, 4, 5]


def main():
    OUT.mkdir(exist_ok=True)

    for scenario in sorted(Path("scenarios").glob("*.txt")):
        for backend, model in MODELS:
            model_slug = model.replace("/", "_")
            for seed in SEEDS:
                stem = f"{scenario.stem}_{model_slug}_seed{seed}"
                out_path = OUT / f"{stem}.json"

                # Skip runs that already have a trace on disk.
                if out_path.exists():
                    print(f"[skip] {out_path}")
                    continue

                subprocess.run([
                    "tracs_gen", str(scenario),
                    "--output", str(out_path),
                    "--backend", backend,
                    "--coder-model", model,
                    "--judge-model", model,
                    "--seed", str(seed),
                    "--coder-temperature", "0.3",
                    "--max-iterations", "10",
                ], check=False)
