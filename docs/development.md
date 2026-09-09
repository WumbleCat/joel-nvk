# Development

The project is managed with [uv](https://docs.astral.sh/uv/). uv owns the Python
interpreter, the virtualenv (`.venv/`) and the lockfile (`uv.lock`), so there is
nothing to activate and no `pip install` step.

## Install uv

| Platform | Command |
| --- | --- |
| macOS / Linux | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Windows (PowerShell) | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| Homebrew | `brew install uv` |
| pipx | `pipx install uv` |

Check it: `uv --version`.

## Set up the project

```bash
git clone <repo-url> joel-nvk
cd joel-nvk
./scripts/setup.sh          # installs uv if missing, syncs the env, creates .env
```

Or do the two steps by hand:

```bash
uv sync                     # creates .venv from uv.lock, installs the dev group
cp .env.example .env        # Windows: copy .env.example .env
```

`uv sync` also downloads a matching Python (`requires-python = ">=3.10"`) if the
machine does not have one — no system Python setup required.

## Running things

Prefix any command with `uv run`. It resolves the environment first, so it is
always running against the locked dependencies and your current `src/`:

```bash
uv run joel-nvk config                    # the CLI
uv run python -m joel_nvk.cli config      # same thing without the entry point
uv run scripts/run_experiment.py --env dev --name baseline
uv run pytest
uv run jupyter lab                        # notebooks/
```

Activating `.venv` manually (`source .venv/bin/activate`,
`.venv\Scripts\activate` on Windows) works too, but `uv run` is preferred: it
cannot go stale against the lockfile.

## Everyday commands

| Command | uv equivalent | What it does |
| --- | --- | --- |
| `make sync` | `uv sync` | Environment matches `uv.lock` |
| `make test` | `uv run pytest` | Test suite |
| `make cov` | `uv run pytest --cov=joel_nvk` | Tests with coverage |
| `make lint` | `uv run ruff check .` | Lint |
| `make format` | `uv run ruff format . && uv run ruff check --fix .` | Format and autofix |
| `make typecheck` | `uv run mypy` | Type check |
| `make clean` | — | Remove caches and build artefacts |

`make` is optional; the uv commands are the source of truth and work identically
on Windows without `make` installed.

## Dependencies

```bash
uv add torch transformers          # runtime dependency
uv add --dev pytest-xdist          # dev-group dependency
uv remove <package>
uv lock --upgrade-package numpy    # bump one package
uv sync                            # apply the lockfile to .venv
```

`uv add` edits `pyproject.toml`, updates `uv.lock` and syncs `.venv` in one step.
**Commit `uv.lock`** — it is what makes a run reproducible on another machine, and
`uv sync --frozen` on a cluster node installs exactly those versions.

Runtime dependencies go in `[project.dependencies]`; anything only needed for
development goes in the `dev` dependency group (`uv add --dev`). CI and cluster
jobs that do not need the tooling use `uv sync --no-dev`.

## Tests

- `tests/unit/` — no I/O, no network, fast enough to run on every save.
- `tests/integration/` — the CLI and anything crossing a boundary.
- Shared fixtures in `tests/conftest.py`; use `tmp_outputs` rather than writing
  into the repo's real `outputs/`.

Mark slow tests with `@pytest.mark.slow` and skip them locally with
`uv run pytest -m "not slow"`.

## GPU

`uv sync --group ml` installs the CUDA 12.6 build of torch on Windows and Linux
(from PyTorch's own index — PyPI's Windows wheel is CPU-only). Check it took:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A driver older than the CUDA build refuses to load it (`nvidia-smi` shows the
newest CUDA the driver supports); on such a machine change the index in
`pyproject.toml` (`pytorch-cu126` → an older `cuXXX`) and `uv lock`. On macOS
there is no CUDA build and the PyPI wheel is used as is.

Run stages on CPU only for tiny sanity checks: an unmerged LoRA on fp32 CPU
decodes at ~1 token/s on this laptop, which makes even the smoke config take
hours.

## Troubleshooting

**`invalid peer certificate: UnknownIssuer` when uv fetches packages.** Something
on the machine (corporate proxy, VPN or AV TLS inspection) re-signs HTTPS traffic
with a certificate uv's bundled root store does not know. Tell uv to use the
system certificate store:

```bash
uv sync --native-tls
export UV_NATIVE_TLS=1     # Windows PowerShell: $env:UV_NATIVE_TLS = "1"
```

Setting the environment variable once per shell (or in your profile) makes every
later `uv` command work without the flag. This machine needs it.

**`There is not enough space on the disk` while installing torch.** uv unpacks
wheels in its cache under `%LOCALAPPDATA%\uv\cache` on C:, and the CUDA torch
wheel is 2.4 GB compressed. Point the cache at a drive with room — a
project-local directory works and is gitignored:

```powershell
$env:UV_CACHE_DIR = "E:\sourcecode\joel-nvk\.uv-cache"   # per shell, or in your profile
uv sync --group ml
```

**`VIRTUAL_ENV=... does not match the project environment path`.** A different
virtualenv is active in the shell; uv ignores it and uses `.venv` anyway, which is
what you want. Deactivate the other env to silence the warning, or pass `--active`
to deliberately target it.

**uv cannot find a suitable Python.** `uv python install 3.11` fetches one; uv
does not need a system Python.

## Adding a module

New code goes under the subpackage matching its job (`core`, `models`, `data`,
`utils`) and keeps the dependency direction in
[architecture.md](architecture.md). Add the matching test in the same commit.

## Notebooks

`notebooks/` is for exploration only. Start Jupyter with `uv run jupyter lab` so
the kernel sees the project environment. Once something works, move it into
`src/joel_nvk/` and import it back — notebooks are not reviewed and are never the
source of a number in the paper.
