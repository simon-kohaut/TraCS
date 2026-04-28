# TraCS - Trajectory Compliance-Shaping

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
tracs_gen scenarios/1_halt.txt --output "run/out/halt_output.txt" --backend local --max-iterations 5
```

Or run all scenarios sequentially:

```shell
bash run/all.sh
```

## TODOs LLM pipeline

- Changes needed:
  - left_primary -> primary_positive_buffer (primary_positive_offset / primary_left_buffer)
  - Removed service and unclassified road
  - add vehicle and pedestrian as features

- Probabilistic predicates
- scenarios:
    - vielleicht irgendwann: andere autos explizit modellieren -> cool weil first oder
