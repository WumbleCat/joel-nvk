---
name: training-pipeline
description: How the four training arms (sft, self_sft, iter_sft, grpo) are implemented and kept comparable — shared trainer contract, on-policy data generation, checkpointing for the KL ladder, and the invariants that must hold across arms. Use when writing or modifying training code, adding an arm, changing checkpointing or optimiser settings, or debugging a run that will not reach its KL targets.
---

# Training pipeline

## Shared contract

Every arm implements the same interface so that differences between arms are the
*method*, never the plumbing:

```python
def train(base_model, task, config, seed) -> Iterator[Checkpoint]
```

- Yields checkpoints on a schedule the caller controls; never decides on its own
  when to stop.
- Never evaluates. Evaluation is a separate pass over checkpoints (see the
  `evaluation` skill) so that every arm is measured by identical code.
- Never touches `outputs/results/`. It writes checkpoints and a step-level
  training log; metrics come later.

Anything shared — optimiser construction, LR schedule, precision, gradient
clipping, batch construction, padding — lives in one place and is called by all
four arms. If an arm needs a different value, it comes from config, not from a
fork of the code.

## The four arms

### `sft`
Cross-entropy on the fixed target dataset. Off-policy, static data. The baseline
everything else is compared against.

### `self_sft`
1. Sample `k` completions per prompt from the **base** model at a fixed
   temperature.
2. Filter by the task's correctness/reward criterion.
3. SFT on the survivors.

Data is on-policy w.r.t. the base model; the objective is identical to `sft`.

### `iter_sft`
`self_sft` repeated for `R` rounds, regenerating data from the **current** policy
at the start of each round. Data freshness increases with round; the objective
stays supervised. Log which round each checkpoint belongs to — round boundaries
show up as discontinuities in the KL trajectory and must be visible in plots.

### `grpo`
Group-relative policy optimisation: sample a group of `G` completions per prompt,
score them, advantage-normalise **within the group**, and take a policy-gradient
step.

- If a KL penalty to the base model is enabled, its coefficient is a swept
  variable, not a fixed constant — it directly controls the quantity being
  matched, so it must be reported alongside every GRPO result.
- Record the mean and variance of within-group rewards. Collapse to zero variance
  means no learning signal, and the run is dead regardless of loss curves.

## Keeping arms comparable

These must be identical across arms unless the config says otherwise, and any
difference must be stated in the results file:

- Base model checkpoint, tokenizer, precision, attention implementation.
- Adaptation surface (full fine-tune vs LoRA rank / target modules).
- Optimiser, LR schedule shape, warmup, weight decay, gradient clipping.
- Maximum sequence length and truncation side.
- Prompt formatting and chat template — including whether the loss is masked on
  the prompt tokens. Mask consistently; an unmasked prompt in one arm alone will
  fake a method effect.

## Checkpointing for the KL ladder

The ladder can only be matched if checkpoints exist near each rung.

- Checkpoint on a **log-spaced step schedule**, dense early where KL moves fast.
- Track a cheap online KL proxy during training and **trigger an extra checkpoint
  when the proxy crosses a rung**. This is the difference between a real match and
  an interpolated one.
- Keep training past the top rung — an arm that stops early leaves the high-KL
  end of the sweep empty for that method.
- Each checkpoint directory carries a manifest: step, wall-clock, tokens seen,
  optimiser state presence, config hash, git commit, seed.

## Determinism and seeding

Seed Python, NumPy and Torch from the single run seed via
`joel_nvk.utils` seeding, and seed the **generation** RNG separately and
explicitly — on-policy arms depend on sampling, and reusing the training RNG makes
data generation silently sensitive to unrelated code changes.

## Logging during training

Per logging interval: loss, LR, grad norm, tokens/s, online KL proxy, and — for
on-policy arms — accept rate after filtering, mean reward, reward variance, and
completion length distribution. Length is not cosmetic: it is a leading indicator
of the format-drift confound.

## Failure modes to watch

| Symptom | Likely cause |
| --- | --- |
| KL plateaus below the lowest rung | LR too small, or KL penalty too strong in `grpo` |
| KL explodes past the ladder in a few steps | LR too high, or no gradient clipping; the ladder has no usable low rungs |
| `self_sft` accept rate near 0 | Filter too strict or temperature too low; the arm is training on almost nothing |
| GRPO reward variance ≈ 0 | Group size too small or task too easy/hard; no learning signal |
| Loss fine, completions degenerate | Reward hacking or template mismatch — read raw samples |
