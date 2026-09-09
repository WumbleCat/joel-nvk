# The MATH → MMLU forgetting pipeline

A single-arm, single-seed run that exercises every piece of machinery the real
study needs: fine-tune, evaluate, fine-tune again, re-evaluate, and measure how
far the model moved. It answers "does the plumbing work and produce sane
numbers?" — not "which method forgets less".

## What it does

```
Qwen base
   │  LoRA SFT on MATH (train split)
   ▼
LoRA(MATH)  ──► accuracy on MATH levels 3-5, accuracy on MMLU
   │  continue the SAME adapter on MMLU
   ▼
LoRA(MATH→MMLU) ──► accuracy on MATH levels 3-5 again  ⇒ forgetting
   │
   └─► exact token-level KL between all three states, on both axes
```

The base model is evaluated too, so the MATH score after MMLU training can be
read against both where it started and where it got to.

## Stages

| Stage | Does |
| --- | --- |
| `prepare` | Builds and hashes the frozen train/eval sets, checks MATH train/eval disjointness, writes the run manifest |
| `baseline` | Evaluates the base model on MATH 3-5 and MMLU |
| `train_math` | LoRA SFT on MATH → adapter `math` |
| `eval_math` | Evaluates the `math` adapter on both tasks |
| `train_mmlu` | Continues the same adapter on MMLU → adapter `mmlu` |
| `eval_mmlu` | Evaluates the `mmlu` adapter on both tasks — the forgetting measurement |
| `kl` | Exact token-level KL for `math‖base`, `mmlu‖base`, `mmlu‖math` on both axes |

Stages run in this order regardless of the order you pass them, and are
resumable: `--run-id <existing>` picks the adapters and frozen sets back up.

## Install and run

The ML stack is an opt-in dependency group, so a docs-only checkout stays small:

```bash
uv sync --group ml        # torch, transformers, peft, datasets, accelerate
```

Then:

```bash
uv run scripts/run_pipeline.py --env smoke      # ~16 examples per stage; does it work at all
uv run scripts/run_pipeline.py --env pilot      # a first pass with meaningful sample sizes
uv run scripts/summarize_run.py --latest        # the accuracy / forgetting / KL tables
```

Re-run a single stage against an existing run:

```bash
uv run scripts/run_pipeline.py --env smoke --stages kl --run-id pipeline-Qwen2.5-0.5B-Instruct-s42-...
```

The first run downloads the model (~1 GB for Qwen2.5-0.5B-Instruct) and the two
datasets; both are cached by Hugging Face afterwards. `smoke` runs on CPU in a
few minutes; `pilot` wants a GPU.

## Configs

| Config | Model | Sizes | For |
| --- | --- | --- | --- |
| `smoke.yaml` | Qwen2.5-0.5B-Instruct | 16 train / 8 eval, 4 KL steps | Does the pipeline execute end to end |
| `default.yaml` | Qwen2.5-0.5B-Instruct | 512 train / 200 eval | The baseline everything else overrides |
| `pilot.yaml` | Qwen2.5-1.5B-Instruct | 2000 train / 500 eval | Numbers worth looking at |

## Outputs

```
outputs/results/<run_id>.jsonl              one record per stage
outputs/results/<run_id>.predictions.jsonl  every completion, graded
outputs/results/<run_id>.manifest.json      commit, config, data hashes, environment
outputs/checkpoints/<run_id>/math|mmlu/     the LoRA adapters
outputs/logs/pipeline-<env>.log             the run log
data/processed/<name>-<hash>.jsonl          the frozen train/eval sets
```

Result records are append-only. `summarize_run.py` reads the `.jsonl` and nothing
else, so a printed number is a recorded number.

## How the numbers are produced

**Accuracy.** Greedy decoding for every state, one shared extraction path.
MATH is graded on the last `\boxed{...}` after LaTeX normalisation, with a
fraction/decimal fallback; MMLU on the generated option letter. The parse failure
rate and mean completion length are reported next to every accuracy — a score
that moves because the format drifted is not forgetting.

**KL.** Continuations are generated **once by the base model** and frozen, then
scored under all three states, so every state is measured on identical tokens.
The estimator is the exact token-level KL summed over the full vocabulary, in
float32, averaged over continuation tokens only, in the fixed direction
`KL(policy ‖ reference)`. A bootstrap standard error over prompts travels with
every value.

Both axes are computed and never mixed:

- `kl_old` — on MATH prompts, where forgetting is measured.
- `kl_new` — on MMLU prompts, where the second training happened.

The three states share one set of base weights; the base distribution comes from
disabling the adapters, so the probe never holds three models in memory.

## What this run cannot tell you

- One seed, one drift point, one arm. No claim about methods survives from here.
- MMLU trained to convergence at whatever KL it lands on — nothing is
  drift-matched yet. Matching is what the real sweep adds.
- Small evaluation sets: the smoke config's accuracies are noise.

See [`.claude/skills/experiment-design/SKILL.md`](../.claude/skills/experiment-design/SKILL.md)
for what turns this into an experiment.
