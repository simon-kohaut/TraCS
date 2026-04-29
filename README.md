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
GOOGLE_API_KEY=<your_google_ai_studio_api_key>
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

Each run produces a single JSON object like this (values abbreviated for readability):

```json
{
  "parameters": {
    "scenario": "1_halt",
    "natural_language_rule": "... full text of the input regulation ...",
    "system_coder": "... system prompt for the coder agent ...",
    "system_judge": "... system prompt for the judge agent ...",
    "backend": "openai",
    "coder_model": "gpt-5.4-mini",
    "judge_model": "gpt-5.4-mini",
    "max_iterations": 5,
    "seed": 1,
    "coder_temperature": 0.3
  },
  "iterations": [
    {
      "program": "... Resin program produced in this iteration ...",
      "judge_passed": false,
      "critique": "... judge critique text ...",
      "compiled": false,
      "compiler_errors": "... compiler error messages ..."
    },
    {
      "program": "...",
      "judge_passed": true,
      "critique": "",
      "compiled": true,
      "compiler_errors": ""
    }
  ],
  "accepted": true,
  "result": "... accepted program string ..."
}
```

### Showing all traces

```shell
for f in output/*.json; do tracs_show "$f"; done
```

## TODOs LLM pipeline

- Changes needed:
  - Removed service and unclassified road
  - add vehicle and pedestrian as features

- Probabilistic predicates
- scenarios:
    - vielleicht irgendwann: andere autos explizit modellieren -> cool weil first oder
