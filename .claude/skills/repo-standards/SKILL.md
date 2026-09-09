---
name: repo-standards
description: Conventions for this repo — where code goes, how configs and paths work, naming for runs and artefacts, and the import/dependency rules. Use before adding a module, script, config, or output file, and when deciding whether something belongs in src/, scripts/, or notebooks/.
---

# Repo standards

This is an ML research repo studying **catastrophic forgetting under matched KL
drift**. Code is judged on whether a result can be reproduced and trusted, not on
cleverness.

## Where things go

| Put it here | When |
| --- | --- |
| `src/joel_nvk/core/` | Experiment logic: KL matching, run orchestration, hypothesis-level code |
| `src/joel_nvk/models/` | Model loading, adapters, the four training methods |
| `src/joel_nvk/data/` | Dataset loading/splitting for the new task and the old-task probes |
| `src/joel_nvk/utils/` | Config, logging, paths, seeding. Depends on nothing else in the package |
| `scripts/` | Thin CLI wrappers: argument parsing plus a call into `src/`. No logic worth testing |
| `notebooks/` | Exploration and figure drafting. Never the source of a number in the paper |
| `tests/` | `unit/` for pure logic (KL math, matching, metrics), `integration/` for CLI and pipeline wiring |

Dependency direction is one-way: `cli.py` → `core` → `models`/`data` → `utils`.
If `utils` needs to import from `models`, the code is in the wrong place.

**Anything that produces a number in the paper lives in `src/` and has a test.**
A notebook cell that computes a headline metric is a bug — move it into the
package and import it back into the notebook.

## Configs

- `configs/default.yaml` holds every key with a working default.
- `dev.yaml` / `production.yaml` contain only overrides and are deep-merged over
  the default by `joel_nvk.utils.config.load_config`.
- Experiment configs are data, not code: a run must be fully described by
  `(config, seed, git commit)`. No magic constants in function bodies.
- Secrets go in `.env` (see `.env.example`), never in a YAML file.

## Paths

Always resolve through `joel_nvk.utils.paths` (`data_dir()`, `outputs_dir()`).
Never hardcode a relative path — scripts get launched from the repo root, from
`scripts/`, and from a scheduler, and all three must behave identically.

## Naming

- **Run id**: `{method}-kl{target}-s{seed}-{YYYYMMDD-HHMMSS}`, e.g.
  `grpo-kl0.15-s1-20260909-142233`. It is the filename stem for the logs,
  results, checkpoints and figures belonging to that run.
- **Methods**: `sft`, `self_sft`, `iter_sft`, `grpo` — these exact strings in
  configs, filenames, dataframe columns and plot legends. One vocabulary
  everywhere.
- **KL axes**: `kl_new` (measured on new-task prompts) and `kl_old` (measured on
  old-task prompts). Never write a bare `kl` — the measurement-axis hypothesis
  turns on keeping these two apart.

## Outputs

`outputs/` is untracked. `logs/` for run logs, `results/` for per-run JSON/JSONL
metrics, `figures/` for plots. Results files are append-only records of what
happened — never hand-edit one to fix a number.

## Running things

The project is uv-managed. Every command goes through `uv run` — `uv run pytest`,
`uv run joel-nvk config`, `uv run scripts/run_experiment.py ...` — so it executes
against the locked environment. Dependencies are added with `uv add` /
`uv add --dev`, never a bare `pip install`, and `uv.lock` is committed. See
`docs/development.md`.

## Style

- `ruff` and `mypy` clean (`uv run ruff check .`, `uv run mypy`).
- Type-annotate anything crossing a module boundary.
- Comments explain *why* a choice was made — especially statistical or numerical
  ones — not what the line does.
