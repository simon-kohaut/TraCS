# TraCS - Trajectory Compliance-Shaping

## Install

```shell
pip install -e .
```

## Usage

Create a `.env` file in the root of the project with the following content:

```env
ANTHROPIC_API_KEY=<your_anthropic_api_key>
OPENAI_API_KEY=<your_openai_api_key>
LOCAL_MODEL_KEY=<your_local_model_key from vLLM or similar servers>
```

Possibly, start your local model and make it available locally. For example, if using vLLM:

```shell
ssh -L 54112:localhost:54112 dgx2-b
```

Then, run the following command to execute a scenario:

```shell
tracs_gen scenarios/1_halt.txt --output run/out/1_halt.json --backend local --max-iterations 5 --verbose

tracs_gen scenarios/1_halt.txt --output run/out/1_halt.json --backend openai --coder-model gpt-5.4-mini --judge-model gpt-5.4-mini --max-iterations 5 --verbose
```

Or run all scenarios sequentially:

```shell
bash run/all.sh
```

## Output

Each run produces a trace derived from `--output`: `<stem>.json`.

### JSON trace schema

| Field | Description |
|---|---|
| `parameters.scenario` | Stem of the input regulation file |
| `parameters.natural_language_rule` | Full text of the input regulation |
| `parameters.system_coder` | System prompt used for the coder agent |
| `parameters.system_judge` | System prompt used for the judge agent |
| `parameters.backend` | LLM backend used |
| `parameters.coder_model` | Model name for the coder |
| `parameters.judge_model` | Model name for the judge |
| `parameters.max_iterations` | Maximum refinement iterations |
| `parameters.seed` | Random seed |
| `parameters.coder_temperature` | Sampling temperature for the coder |
| `iterations[].program` | Resin program produced in this iteration |
| `iterations[].judge_passed` | Whether the judge accepted the program |
| `iterations[].critique` | Judge critique text |
| `iterations[].compiled` | Whether the Resin compiler accepted the program |
| `iterations[].compiler_errors` | Compiler error messages (empty string if OK) |
| `accepted` | Whether the pipeline converged |
| `result` | Accepted program string, or `null` on failure |

### Showing all traces

```shell
for f in output/*.json; do tracs_show "$f"; done
```

## TODOs LLM pipeline

- Changes needed:
  - left_primary -> primary_positive_buffer (primary_positive_offset / primary_left_buffer)
  - Removed service and unclassified road
  - add vehicle and pedestrian as features

- Probabilistic predicates
- scenarios:
    - vielleicht irgendwann: andere autos explizit modellieren -> cool weil first oder
