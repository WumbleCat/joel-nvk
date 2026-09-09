# Architecture

## Layout

```
src/joel_nvk/
├── core/     domain logic — the work this project exists to do
├── models/   model definitions, training and inference
├── data/     loading, preprocessing, splitting
├── utils/    config, logging, paths
└── cli.py    the `joel-nvk` command
```

## Dependency direction

`cli.py` → `core` → `models` / `data` → `utils`

`utils` depends on nothing else in the package, and nothing imports `cli.py`.
Keeping the arrows one-way is what lets `core` be tested without touching the
filesystem or the CLI.

## Configuration

`configs/default.yaml` holds every key. `dev.yaml` and `production.yaml` contain
only the keys they change, and are deep-merged over the default by
`joel_nvk.utils.config.load_config`. Secrets never live in these files — they go
in `.env` (see `.env.example`).

## Filesystem contract

| Path              | Written by            | Tracked by git |
| ----------------- | --------------------- | -------------- |
| `data/raw/`       | `scripts/download_data.py` | no        |
| `data/processed/` | `joel_nvk.data`       | no             |
| `outputs/logs/`   | `joel_nvk.utils.logging` | no          |
| `outputs/figures/`| plotting code         | no             |
| `outputs/results/`| `scripts/run_experiment.py` | no        |

Paths are resolved from the repo root by `joel_nvk.utils.paths`, so scripts
behave the same regardless of the working directory they were launched from.
