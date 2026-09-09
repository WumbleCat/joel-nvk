# joel-nvk

Research code for a study of **catastrophic forgetting under matched KL drift**:
if two models have moved equally far from the base model, do different
post-training methods still cause different amounts of forgetting?

Prior work compares SFT against on-policy/RL post-training at equal *training
budget* and finds RL forgets less — a comparison that confounds **how far the
model moved** with **how it was trained**. This repo controls drift and varies
method across four arms (`sft`, `self_sft`, `iter_sft`, `grpo`) and two KL
measurement axes (`kl_new`, `kl_old`). See
[`.claude/skills/experiment-design/SKILL.md`](.claude/skills/experiment-design/SKILL.md)
for the hypotheses and the sweep protocol.

Managed with [uv](https://docs.astral.sh/uv/) — uv owns the interpreter, the
virtualenv and `uv.lock`, so there is nothing to activate.

## Setup

```bash
# 1. install uv (skip if you have it)
curl -LsSf https://astral.sh/uv/install.sh | sh                                    # macOS/Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 2. set the project up
./scripts/setup.sh        # syncs the env from uv.lock and creates .env
```

By hand instead of the script:

```bash
uv sync                   # creates .venv, installs the project + dev group
cp .env.example .env      # Windows: copy .env.example .env
```

## Quick start

```bash
uv run joel-nvk config                            # print the resolved configuration
uv run pytest
```

Run the MATH -> MMLU forgetting pipeline (see [docs/pipeline.md](docs/pipeline.md)):

```bash
uv sync --group ml                                # torch, transformers, peft, datasets
uv run scripts/run_pipeline.py --env smoke        # every stage, tiny sizes
uv run scripts/summarize_run.py --latest          # accuracy / forgetting / KL tables
```

## Repository structure

```
joel-nvk/
├── README.md                    this file
├── LICENSE                      MIT
├── pyproject.toml               project metadata, deps, ruff/mypy/pytest config
├── uv.lock                      the resolved environment — committed, reproducible
├── Makefile                     thin wrappers over the uv commands
├── .env.example                 template for .env (secrets; .env is untracked)
│
├── src/joel_nvk/                the package — all importable code
│   ├── __init__.py
│   ├── cli.py                   the `joel-nvk` entry point
│   ├── core/                    pipeline, evaluation, scoring, KL, reporting
│   ├── models/                  model/adapter loading and LoRA training
│   ├── data/                    MATH and MMLU loading, prompts, collation
│   └── utils/                   config, logging, paths, seeding
│
├── tests/
│   ├── conftest.py              shared fixtures (project_root, configs_dir, tmp_outputs)
│   ├── unit/                    pure logic — no I/O, no network, fast
│   └── integration/             the CLI and anything crossing a boundary
│
├── configs/
│   ├── default.yaml             every key, with a working default
│   ├── dev.yaml                 overrides only: loud logs, small runs
│   ├── smoke.yaml               tiny sizes: does the pipeline run at all
│   ├── pilot.yaml               a first pass with meaningful sample sizes
│   └── production.yaml          overrides only: quiet logs, full parallelism
│
├── scripts/                     thin CLI wrappers; no logic worth testing
│   ├── setup.sh                 install uv, sync the env, seed .env
│   ├── run_pipeline.py          the MATH -> MMLU forgetting run
│   ├── summarize_run.py         accuracy / forgetting / KL tables for a run
│   ├── run_experiment.py        generic single-run entry point
│   └── download_data.py         populate data/raw/
│
├── docs/
│   ├── architecture.md          module boundaries, dependency direction, fs contract
│   ├── development.md           installing uv, setup, dependencies, tests
│   ├── usage.md                 CLI, scripts, configuration, cluster/CI runs
│   └── pipeline.md              the MATH -> MMLU run, stage by stage
│
├── notebooks/
│   └── exploration.ipynb        exploration only — never the source of a paper number
│
├── data/                        (contents untracked)
│   ├── README.md                what lives here and where it came from
│   ├── raw/                     downloaded/received data, never edited in place
│   └── processed/               derived from raw/ by code in this repo
│
├── outputs/                     (contents untracked; skeleton kept via .gitkeep)
│   ├── logs/                    per-run logs
│   ├── figures/                 plots
│   └── results/                 per-run metrics as JSON/JSONL
│
└── .claude/skills/              project skills that steer work in this repo
```

### What goes where

| Directory | Rule |
| --- | --- |
| `src/joel_nvk/` | Anything that produces a number in the paper lives here and has a test |
| `scripts/` | Argument parsing plus a call into `src/` — nothing more |
| `notebooks/` | Scratch only; once it works, move it into `src/` and import it back |
| `configs/` | A run is fully described by `(config, seed, git commit)` — no magic constants in code |
| `data/`, `outputs/` | Untracked. Paths resolve through `joel_nvk.utils.paths`, never hardcoded |

Dependency direction inside the package is one-way:
`cli.py` → `core` → `models`/`data` → `utils`. `utils` imports nothing else in
the package.

### Naming conventions

- **Run id**: `{method}-kl{target}-s{seed}-{YYYYMMDD-HHMMSS}` — the filename stem
  for that run's logs, results, checkpoints and figures.
- **Methods**: `sft`, `self_sft`, `iter_sft`, `grpo` — these exact strings in
  configs, filenames, dataframe columns and plot legends.
- **KL axes**: `kl_new` (new-task prompts) and `kl_old` (old-task prompts). Never
  a bare `kl` — the measurement-axis hypothesis turns on keeping them apart.

## Skills

`.claude/skills/` holds the project's working guidance. Claude Code loads these
automatically; they are also worth reading directly.

| Skill | Covers |
| --- | --- |
| [repo-standards](.claude/skills/repo-standards/SKILL.md) | Where code goes, configs, paths, naming, running via `uv run` |
| [experiment-design](.claude/skills/experiment-design/SKILL.md) | The three hypotheses, the four arms, the KL ladder, confounds |
| [training-pipeline](.claude/skills/training-pipeline/SKILL.md) | Shared trainer contract, on-policy data generation, checkpointing |
| [evaluation](.claude/skills/evaluation/SKILL.md) | KL estimators on both axes, forgetting metrics, decoding rules |
| [statistical-analysis](.claude/skills/statistical-analysis/SKILL.md) | Matched comparisons, the H1/H2 regression, the H3 axis test |
| [reproducibility](.claude/skills/reproducibility/SKILL.md) | Run manifests, seeding, frozen data hashes, `uv.lock` discipline |
| [paper-writing](.claude/skills/paper-writing/SKILL.md) | Framing the contribution, structure, calibrated hedging, limitations |
| [code-review](.claude/skills/code-review/SKILL.md) | Catching silently wrong numbers: KL correctness, arm asymmetry, leakage |

## Common commands

| Command | What it does |
| --- | --- |
| `uv sync` | Make `.venv` match `uv.lock` |
| `uv run joel-nvk config` | Print the resolved configuration |
| `uv run pytest` | Test suite |
| `uv run ruff check .` | Lint |
| `uv run mypy` | Type check |
| `uv add <pkg>` / `uv add --dev <pkg>` | Add a dependency and update the lock |

`make sync`, `make test`, `make lint`, `make format`, `make typecheck` wrap the
same commands. Full details in [docs/development.md](docs/development.md); the
forgetting pipeline is documented in [docs/pipeline.md](docs/pipeline.md).
