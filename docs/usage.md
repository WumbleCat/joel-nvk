# Usage

Everything runs through `uv run`, which syncs the environment before executing.
See [development.md](development.md) for installing uv and setting the project up.

## CLI

```bash
uv run joel-nvk --help
uv run joel-nvk config                    # print the resolved configuration
uv run joel-nvk --env production config   # ...with production.yaml merged in
```

The same thing as a module, if you prefer:

```bash
uv run python -m joel_nvk.cli config
```

## The forgetting pipeline

The MATH -> MMLU run, in full, is described in [pipeline.md](pipeline.md):

```bash
uv sync --group ml                              # one-off: torch, transformers, peft, datasets
uv run scripts/run_pipeline.py --env smoke      # every stage, tiny sizes
uv run scripts/summarize_run.py --latest        # accuracy / forgetting / KL tables
```

## Scripts

```bash
uv run scripts/download_data.py            # populate data/raw/
uv run scripts/run_experiment.py --env dev --name baseline
```

`run_experiment.py` writes `outputs/logs/<run-id>.log` and
`outputs/results/<run-id>.json`, where `<run-id>` is `<name>-<timestamp>`.

## From Python

```python
from joel_nvk.utils import load_config, setup_logging

config = load_config("dev")
setup_logging(config["logging"]["level"])
```

Run such a file with `uv run path/to/file.py`.

## Configuration and secrets

Choose a config with `--env` (any `configs/<name>.yaml`). Secrets come from
`.env`, created from `.env.example` by `scripts/setup.sh`, and never committed.

## On a cluster or in CI

```bash
uv sync --frozen --no-dev      # exactly the locked versions, no dev tooling
uv run --no-sync scripts/run_experiment.py --env production --name sweep
```

`--frozen` fails rather than silently re-resolving if `uv.lock` is out of date,
which is what you want for a run whose numbers go in the paper.
