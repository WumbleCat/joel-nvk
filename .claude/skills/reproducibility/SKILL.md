---
name: reproducibility
description: What every run must record so a number can be traced back to the code, config, data and hardware that produced it — run manifests, seeding, frozen data hashes, checkpoint provenance, and the rules for reruns. Use when launching runs, adding anything that writes to outputs/, changing a data or eval set, or trying to reproduce or explain a past number.
---

# Reproducibility

A claim in this project is only as good as the ability to regenerate it. The
target: **any number in the paper can be traced to a commit, a config, a seed and
a data hash, and rerunning that triple reproduces it within stated noise.**

## Run manifest

Every run writes `outputs/results/{run_id}.manifest.json` **before** training
starts:

```
run_id, method, seed, target_kl,
git_commit, git_dirty (bool), git_branch,
config_resolved (the fully merged dict, not the file name), config_hash,
base_model_id_and_revision, tokenizer_revision,
dataset_hashes {new_task, old_task_suite, kl_prompt_sets},
python_version, package_versions (torch, transformers, trl, numpy, ...),
gpu_model, gpu_count, precision, cuda_version,
launch_command, started_at
```

`git_dirty = true` is allowed for exploration and **disqualifying for any number
that reaches the paper**. Check it at launch and warn loudly.

## Seeding

- One integer seed per run, in the config, in the run id, in the manifest.
- It seeds Python, NumPy, Torch (CPU and CUDA) and the dataloader workers through
  a single helper.
- Generation gets its **own** derived seed (`seed_gen = hash(seed, round)`), so
  on-policy data generation does not shift when unrelated RNG consumers change.
- Seeds are `1..N` — never "the seed that worked".

## Frozen data

The new-task splits, the old-task suite, and both KL prompt sets are generated
once, written to `data/processed/`, and hashed. The hash goes in every manifest
and every results record.

Regenerating any of them starts a **new hash and a new results file**. Never
silently reuse a filename after changing its contents — that is how two
incompatible KL columns end up in the same plot.

`data/processed/` must be rebuildable from `data/raw/` by
`scripts/download_data.py` plus a documented preprocessing entry point. If a step
was done by hand once, it is not reproducible; write it down as code.

## Determinism

Full bitwise determinism on GPU is not worth its cost here. Instead:

- Deterministic **evaluation**: greedy decoding, fixed prompt order, fixed batch
  size. Two evaluations of the same checkpoint must agree exactly. Batch-size
  dependence in eval is a bug — find it.
- Training is allowed to be non-deterministic across hardware; that is what
  multiple seeds are for. Record the hardware so a failure to reproduce can be
  attributed.
- Never change the eval batch size mid-project without re-running the affected
  evaluations.

## Checkpoints

Each checkpoint directory carries `checkpoint_meta.json`: run id, step, tokens
seen, wall-clock, online KL proxy, round index for `iter_sft`, and the manifest
hash. A checkpoint that cannot name its run is deleted, not guessed at.

Retain: every checkpoint that was matched to a ladder rung, plus enough
neighbours to re-match under a different tolerance. Delete the rest deliberately
and log what was deleted.

## Reruns

- Same commit, config, seed, hardware → expect exact eval agreement; a mismatch is
  a bug.
- Same commit, config, seed, *different* hardware → expect agreement within the
  seed-level noise already measured. State the noise level.
- Any change to the KL estimator, prompt sets, or normalisation invalidates
  previous numbers on that axis. Re-run everything affected; never mix estimator
  versions in one table or figure.

## Environment

The environment is uv-managed and **`uv.lock` is the record**, not a prose list of
versions.

- Every run executes through `uv run`, so it uses the locked environment rather
  than whatever happens to be installed.
- Commit `uv.lock`. Paper runs use `uv sync --frozen` (and `--no-dev` on cluster
  nodes), which fails loudly instead of silently re-resolving.
- Record the lock hash in the manifest alongside the resolved package versions, so
  a number can be traced to an exact dependency set.
- Add dependencies with `uv add` / `uv add --dev` — never a manual `pip install`
  into `.venv`. An unlocked install makes the environment unreproducible from that
  moment on, and the next `uv sync` silently removes it.
- Anything that can change numerics (torch, transformers, trl, the eval harness)
  is upgraded deliberately: `uv lock --upgrade-package <name>`, then re-run the
  affected evaluations. Do not mix results from before and after such an upgrade.
