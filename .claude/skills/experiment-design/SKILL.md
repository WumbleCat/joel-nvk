---
name: experiment-design
description: The matched-KL design for separating drift magnitude from training method in catastrophic forgetting — the three hypotheses, the four arms, the sweep protocol, required controls, and the confounds that invalidate a run. Use when planning an experiment, adding an arm or ablation, changing what gets swept, or judging whether a proposed comparison actually tests the claim.
---

# Experiment design

## The question

If two models have moved **equally far** from the base model, do different
post-training methods still cause different amounts of forgetting?

Prior work compares SFT and RL-style post-training at equal *training budget* and
observes that RL forgets less. That comparison confounds **how far the model
moved** with **how it was trained**. This design controls drift and varies method.

## The three hypotheses

| | Hypothesis | Prediction |
| --- | --- | --- |
| **H1** | **Drift-magnitude** — methods forget less only because they move less | At matched KL, forgetting is equal across methods; the method term vanishes once KL is conditioned on |
| **H2** | **Data-source** — training on the model's own fresh, on-policy outputs is inherently protective | At matched KL, `self_sft` / `iter_sft` / `grpo` still forget less than `sft`; the method term survives conditioning on KL |
| **H3** | **Measurement-axis** — prior work measures KL on the wrong distribution | `kl_old` predicts forgetting better than `kl_new`; matching on `kl_new` leaves a method gap that matching on `kl_old` closes |

H1 and H2 are mutually exclusive on a given axis. H3 can hold alongside either,
and is tested by *which* KL axis is used for matching and for prediction.

## Arms

One base model, one new task, four methods:

- `sft` — supervised fine-tuning on the fixed target dataset (off-policy data).
- `self_sft` — SFT on the base model's own filtered samples (on-policy data, one
  generation round, supervised objective).
- `iter_sft` — repeated self-SFT, regenerating data from the *current* policy
  each round (progressively on-policy, still supervised).
- `grpo` — group-relative policy optimisation against the task reward.

`self_sft` and `iter_sft` are the load-bearing arms: they separate the **data
source** from the **RL objective**. If on-policy data alone explains the effect,
they behave like `grpo`. If the objective matters, they behave like `sft`.

## Matched-KL protocol

This is a **sweep, not a pair of runs**:

1. Fix a ladder of target KL values, e.g. `[0.02, 0.05, 0.1, 0.2, 0.4, 0.8]`.
2. Train every method past the top of the ladder, checkpointing densely.
3. At every checkpoint measure `kl_new` and `kl_old` against the frozen base
   model on fixed held-out prompt sets.
4. For each target, select the checkpoint whose KL is closest, and **record the
   achieved KL** — never assume the target was hit.
5. Evaluate forgetting and new-task performance at those matched checkpoints.

Matching is performed **twice**, once per axis, and both are reported. That is
the H3 test.

## Controls that must hold

- Same base model, tokenizer and precision across all arms.
- Same new task and same target/reward signal, so arms differ only in how they
  consume it.
- Same evaluation suite and decoding settings at every checkpoint.
- Same KL estimator, prompt sets and sample counts for every arm.
- **≥3 seeds per (method, target)** — 5 preferred. A single-seed difference at
  matched KL is not a result.
- Equal or explicitly documented compute. If an arm needs more steps to reach a
  KL rung, that is a finding, not a reason to truncate the ladder.

## Confounds to design against

- **Length and format drift** — on-policy methods can shift output length, moving
  benchmark scores without moving knowledge. Log length distributions; report
  format-normalised scores.
- **KL estimator mismatch** — a sampled KL over generations and an exact
  token-level KL on fixed prompts are different quantities. One estimator per
  axis, used everywhere.
- **Checkpoint granularity** — sparse checkpoints mean the achieved KL misses the
  target and the "match" is fictional. Checkpoint densely near the rungs.
- **Reward hacking in GRPO** — a degenerate policy can score well on the new task
  and look non-forgetting. Read samples at every rung.
- **Old-task contamination** — if the new task overlaps the old-task benchmarks,
  forgetting is unmeasurable. Verify disjointness before running.
- **Capacity confound** — LoRA rank or frozen layers change how drift maps to
  forgetting. Hold the adaptation surface identical across arms, or sweep it as
  an explicit ablation.

## Ablations worth the compute

- KL axis swap (already core: `kl_new` vs `kl_old` matching).
- Data freshness within `iter_sft`: 1 / 2 / 4 regeneration rounds.
- Sampling temperature for on-policy data generation — tests whether it is
  *on-policy-ness* or *diversity* that protects.
- Filtering strictness in `self_sft` — separates data quality from data source.

## What a reportable claim looks like

> At matched `kl_old` of 0.20 ± 0.02, `grpo` and `sft` differ in old-task accuracy
> by 1.3 points (95% bootstrap CI [-0.4, 3.1], 5 seeds) — no reliable gap on this
> axis, consistent with H1.

Method, axis, achieved KL with tolerance, effect size, interval, seed count.
Anything less is not reportable.
