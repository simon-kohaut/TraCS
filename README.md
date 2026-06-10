# TraCS - Trajectory Compliance-Shaping

## Installation and Setup

```shell
pip install -e .
```

To get the backbone outputs:
```shell
mkdir -p data; cd data
gdown --folder <<REDACCTED_GOOGLE_DRIVE_LINK>>
```

To get the scenario files:
```shell
# Assuming you are still in data/, run:
mkdir -p av2; cd av2

curl -LO https://s3.amazonaws.com/argoverse/datasets/av2/motion-forecasting/av2_mf_focal_test_annotations.parquet
curl -LO https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/train.tar
curl -LO https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/val.tar
curl -LO https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/test.tar

# Unpack the scenario files:
tar -xf train.tar; tar -xf val.tar; tar -xf test.tar

# Cleanup the tar files:
rm train.tar val.tar test.tar

# Go back to data/:
cd ../..
```

The final directory structure should look like this:

```
data/
├── av2/
│   ├── train/
│   │   ├── 0000b0f9-99f9-4a1f-a231-5be9e4c523f7/
│   │   │   ├── log_map_archive_0000b0f9-99f9-4a1f-a231-5be9e4c523f7.json
│   │   │   └── scenario_0000b0f9-99f9-4a1f-a231-5be9e4c523f7.parquet
│   │   └── ...
│   ├── val/
│   │   └── ...
│   ├── test/
│   │   └── ...
│   └── av2_mf_focal_test_annotations.parquet
└── av2_backbone_outputs/
    ├── DeMo/
    │   ├── submission_test.parquet
    │   ├── submission_val.parquet
    │   ├── test_embeddings.parquet
    │   ├── val_embeddings.parquet
    │   ├── metrics.parquet
    │   └── scores.txt
    ├── Polaris/
    │   └── ...
    └── QCNet/
        └── ...
```

#### Setup for LLM API Keys and Local Models
Create a `.env` file in the root of the project with the following content:

```env
ANTHROPIC_API_KEY=<your_anthropic_api_key>
OPENAI_API_KEY=<your_openai_api_key>
GOOGLE_API_KEY=<your_google_ai_studio_api_key>
LOCAL_MODEL_KEY=<your_local_model_key from vLLM or similar servers>
```

Possibly, to make your self-hosted model available locally, for example, if using vLLM, run:

```shell
ssh -L 54112:localhost:54112 remote_host
```

## Usage

### Running the Agentic Pipeline

#### Command Overview

| Command | Description |
|---|---|
| `tracs_agents_run` | Run the agentic pipeline on a single scenario |
| `tracs_agents_run_all` | Run all scenarios across all configured models and seeds |
| `tracs_agents_show_trace` | Pretty-print a JSON trace file |
| `tracs_agents_append_judge` | Add judge opinions to an existing trace post-hoc |

Run with `--help` for more details on each command and its arguments.

Each run produces a trace derived from `--output`: `<stem>.json`.

#### Running the pipeline on a scenario

Run the following command to execute a scenario:

```shell
tracs_agents_run scenarios/1_halt.txt --output run/out/1_halt.json \
  --coder-backend anthropic --judge-backend anthropic --max-iterations 5

tracs_agents_run scenarios/1_halt.txt --output run/out/1_halt.json \
  --coder-backend openai --coder-model gpt-5.4-mini \
  --judge-backend openai --judge-model gpt-5.4-mini \
  --max-iterations 5

# Separate backends for coder and judge:
tracs_agents_run scenarios/1_halt.txt --output run/out/1_halt.json \
  --coder-backend local --coder-model Qwen/Qwen3.6-27B \
  --judge-backend anthropic --judge-model claude-opus-4-7 \
  --max-iterations 5
```

Or run all scenarios and models sequentially:

```shell
tracs_agents_run_all
```

The processing speed depends on the LLM inference speed and, importantly, the number of iterations.

#### JSON Trace Schema

Each run produces a single JSON object like this (values abbreviated for readability):

```json
{
  "parameters": {
    "scenario": "1_halt",
    "natural_language_rule": "... full text of the input regulation ...",
    "system_coder": "... system prompt for the coder agent ...",
    "system_judge": "... system prompt for the judge agent ...",
    "coder_backend": "openai",
    "coder_model": "gpt-5.4-mini",
    "judge_backend": "anthropic",
    "judge_model": "claude-opus-4-7",
    "max_iterations": 5,
    "seed": 1,
    "coder_temperature": 0.3
  },
  "iterations": [
    {
      "program": "... Resin program produced in this iteration ...",
      "full_program": "... full program including source signals and target predicates ...",
      "judge_passed": false,
      "critique": "... judge critique text ...",
      "compiled": false,
      "compiler_errors": "... compiler error messages ...",
      "opinions": {
        "human": {"passed": true},
        "claude-opus-4-7": {"passed": false, "critique": "..."}
      }
    },
    {
      "program": "...",
      "full_program": "...",
      "judge_passed": true,
      "critique": "",
      "compiled": true,
      "compiler_errors": "",
      "opinions": {}
    }
  ],
  "accepted": true,
  "final_program": "... accepted program string ...",
  "full_final_program": "..."
}
```

#### Showing All Traces

This will loop over all traces:

```shell
for f in llm_outputs/*.json; do echo "$f:"; tracs_agents_show_trace "$f"; read -p "Press Enter for next..."; echo ; done
```

#### Adding Judge Opinions Post-hoc

Use `tracs_agents_append_judge` to annotate an existing trace with additional judge verdicts.
Opinions are stored per iteration under the `opinions` key, keyed by judge name.

```shell
# Interactive human annotation (y/n/s per iteration)
tracs_agents_append_judge output/1_halt_claude-opus-4-7_seed1.json --judge human

# LLM judge (uses the system_judge prompt already embedded in the trace)
tracs_agents_append_judge output/1_halt_claude-opus-4-7_seed1.json \
  --judge claude-opus-4-7 --backend anthropic
```

To run many:

```shell
for f in llm_outputs/my_traffic_rule/*.json; do tracs_agents_append_judge "$f" --judge human; done
```

After having added human opinions, you can analyze agreement between the human and LLM judges. See `analysis/agent-loop.ipynb` for automated analysis.
