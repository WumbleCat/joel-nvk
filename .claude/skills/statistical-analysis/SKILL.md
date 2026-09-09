---
name: statistical-analysis
description: How results are turned into claims — matched-KL comparisons, the regression that adjudicates H1 vs H2, the axis comparison for H3, bootstrap intervals over seeds, and the analysis errors that would make a null look like an effect. Use when analysing results, choosing a test, making a figure that carries a claim, or interpreting whether a difference between arms is real.
---

# Statistical analysis

The unit of analysis is a **(method, seed, checkpoint)** row from
`outputs/results/*.jsonl`. Analysis code reads those files and nothing else.

## Pre-commit the analysis

Before looking at outcomes, write down: the KL ladder and match tolerance, the
seed count, the primary axis (`kl_old` or `kl_new`), the primary old-task metric,
and the H1/H2/H3 decision rules. Put it in `docs/` and commit it. Every analysis
run after that is either the pre-specified one or is labelled exploratory in the
paper.

## The two complementary analyses

### 1. Matched comparison (direct, assumption-light)

At each ladder rung, take the checkpoint per (method, seed) closest to the target
KL, then compare methods.

- Report the **achieved** KL per arm with its spread. If achieved KLs differ by
  more than the tolerance, the comparison is not matched — say so and fall back to
  the regression.
- Bootstrap over **seeds** (resample seeds, not items) for CIs on the
  method difference. Seeds are the replication unit; items within a benchmark are
  not independent replications of a training run.
- With 3–5 seeds, report effect sizes and intervals. Do not lean on p-values from
  n=3; a CI that includes zero is reported as "no reliable difference", not as
  "no difference".

### 2. Conditional model (uses the whole sweep)

```
forgetting ~ f(kl) + method + (1 | seed)
```

with `f(kl)` flexible (log-KL, or a spline) because forgetting is not linear in
KL. Fit separately with `kl = kl_old` and `kl = kl_new`.

- **H1 supported**: the `method` coefficients are small with intervals covering
  zero once `f(kl)` is included — drift explains the difference.
- **H2 supported**: the on-policy arms keep a reliably negative `method`
  coefficient after conditioning on KL.
- Always report the method effect *before* and *after* conditioning. The shrinkage
  between them is the actual quantity of interest, and it is what prior work
  never reported.

Only compare methods over the **KL range all of them actually cover**. Extrapolating
one arm's curve into a region only another arm reached is the easiest way to
manufacture an effect.

## H3: which axis predicts forgetting

Fit the same model on each axis and compare predictive quality — cross-validated
$R^2$ or held-out log-likelihood, with the **fold split by seed**, not by row.

Report three things:
1. Predictive quality of `kl_old` vs `kl_new`.
2. The residual method effect under `kl_new` matching.
3. The residual method effect under `kl_old` matching.

H3 is supported when `kl_old` predicts better **and** the method gap shrinks
substantially when matching moves from `kl_new` to `kl_old`.

## Multiplicity

There are four methods, several benchmarks, two axes and several rungs. Designate
**one** primary comparison (primary axis, primary rung range, primary old-task
metric) up front. Everything else is secondary and labelled as such. If a family
of comparisons is reported as evidence, adjust within the family (Holm) and say so.

## Figures that carry claims

- **Forgetting vs KL, one line per method, one panel per axis.** This is the main
  figure — H1 is "the lines overlap", H2 is "they separate".
- Plot every seed as a light point behind the mean line. A mean line alone hides
  the variance the claim depends on.
- Use log-x for KL; the interesting structure is at small KL.
- Mark the ladder rungs and the matched checkpoints explicitly.
- Same method→colour mapping in every figure, matching the `sft` / `self_sft` /
  `iter_sft` / `grpo` vocabulary.
- Pair every forgetting plot with the new-task performance plot on the same axis.

## Errors that would fake a result

- Comparing at equal **step** rather than equal KL — that is the confound the
  project exists to remove.
- Bootstrapping over benchmark items instead of seeds: intervals shrink to
  nothing and every difference becomes "significant".
- Dropping a diverged or crashed run without reporting it — selection on outcome.
- Selecting the checkpoint that best fits the story instead of the one nearest the
  target KL.
- Reporting relative forgetting only, where a small base-vs-floor denominator
  inflates the effect.
- Comparing arms at different achieved KLs and calling it "approximately matched".
- Reading a null from an underpowered design as evidence for H1. State the
  smallest effect the design could have detected.
