# TraCS - Trajectory Compliance-Shaping

## Install

```shell
pip install -e .
```

TODO for submission: redact the Google drive link, they cannot easily be made anonymous

To get the backbone outputs:
```shell
cd data
gdown --folder 1Vk8WSlfocli7gVtyPWC-pU3QbbxVCZXm
```

To get the scenario files:
```shell
# Assuming you are still in data/
mkdir av2
cd av2

curl -LO "https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/train.tar"
curl -LO "https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/val.tar"
curl -LO "https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/test.tar"

# Unpack the scenario files:
tar -xf train.tar
tar -xf val.tar
tar -xf test.tar

# Cleanup the tar files:
rm train.tar val.tar test.tar
cd ..
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
│   └── test/
│       └── ...
└── av2_backbone_outputs/
    ├── DeMo/
    │   ├── submission.parquet
    │   └── metrics.parquet
    └── ...
```

## Usage

Create a `.env` file in the root of the project with the following content:

```env
ANTHROPIC_API_KEY=<your_anthropic_api_key>
OPENAI_API_KEY=<your_openai_api_key>
GOOGLE_API_KEY=<your_google_ai_studio_api_key>
LOCAL_MODEL_KEY=<your_local_model_key from vLLM or similar servers>
```

Possibly, start your local model and make it available locally. For example, if using vLLM:

```shell
ssh -L 54112:localhost:54112 dgx2-b
```

Then, run the following command to execute a scenario:

```shell
tracs_gen scenarios/1_halt.txt --output run/out/1_halt.json \
  --coder-backend anthropic --judge-backend anthropic --max-iterations 5 --verbose

tracs_gen scenarios/1_halt.txt --output run/out/1_halt.json \
  --coder-backend openai --coder-model gpt-5.4-mini \
  --judge-backend openai --judge-model gpt-5.4-mini \
  --max-iterations 5 --verbose

# Separate backends for coder and judge:
tracs_gen scenarios/1_halt.txt --output run/out/1_halt.json \
  --coder-backend local --coder-model Qwen/Qwen3.6-27B \
  --judge-backend anthropic --judge-model claude-opus-4-7 \
  --max-iterations 5 --verbose
```

Or run all scenarios sequentially:

```shell
bash run/all.sh
```

## Output

Each run produces a trace derived from `--output`: `<stem>.json`.

### JSON trace schema

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

### Showing all traces

```shell
for f in output/*; do echo "$f:"; tracs_show "$f"; read -p "Press Enter for next..."; echo ; done
```

### Adding judge opinions post-hoc

Use `tracs_add_judge` to annotate an existing trace with additional judge verdicts.
Opinions are stored per iteration under the `opinions` key, keyed by judge name.

```shell
# Interactive human annotation (y/n/s per iteration)
tracs_add_judge output/1_halt_claude-opus-4-7_seed1.json --judge human

# LLM judge (uses the system_judge prompt already embedded in the trace)
tracs_add_judge output/1_halt_claude-opus-4-7_seed1.json \
  --judge claude-opus-4-7 --backend anthropic
```

## TODOs LLM pipeline

- Probabilistic predicates
- scenarios:
    - vielleicht irgendwann: andere autos explizit modellieren und dann:
    ```
    distance(vehicle1) ← source(“…“).
    distance(vehicle2) ← source(“…“).
    is_vehicle(vehicle1).
    safe_car if is_vehicle(X) and distance(X) > 5.
    landscape if safe_car.
    ```
- could make it easier to query multiple LLMs in parallel: https://gemini.google.com/share/3005c3443542
