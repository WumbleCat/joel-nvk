---
name: evaluation
description: How checkpoints are measured — the KL estimators on both axes, the old-task forgetting suite, new-task performance, and the decoding and normalisation rules that keep every arm comparable. Use when writing or changing evaluation code, adding a benchmark, defining a forgetting metric, or debugging scores that move for suspicious reasons.
---

# Evaluation

Evaluation is a **separate pass over checkpoints**, never part of training. One
code path measures every arm, so a difference between arms can never be a
difference in measurement.

## The two KL axes

Both are computed against the **frozen base model** and both are recorded for
every checkpoint. Never collapse them into one number.

| Field | Prompt distribution | Answers |
| --- | --- | --- |
| `kl_new` | Held-out prompts from the **new task** | "How far did the model move where it was trained?" |
| `kl_old` | Prompts from the **old-task** benchmarks | "How far did the model move where forgetting is measured?" |

H3 lives entirely in the gap between these two.

### Estimator rules

- Use **token-level KL on a fixed prompt set with teacher-forced continuations**
  as the primary estimator: it is low-variance and exactly reproducible.
  Sequence-level sampled KL may be reported as a secondary check, never mixed
  into the same column.
- Fixed prompt sets, fixed order, fixed size, frozen at the start of the project
  and stored under `data/processed/` with a content hash. Changing the prompt set
  invalidates every KL number already collected.
- Average over tokens, not sequences, and record the token count. Length drift
  otherwise silently reweights the estimate.
- Compute in float32 regardless of training precision, and record the number of
  prompts and tokens alongside the value so its noise floor is known.
- Report a bootstrap standard error for each KL. A match tolerance narrower than
  the estimator's own noise is a fiction.

## Old-task suite (forgetting)

- Several benchmarks spanning distinct capabilities — knowledge, reasoning,
  instruction-following, safety/refusal behaviour if relevant. A single benchmark
  measures one kind of forgetting and generalises to nothing.
- **Verify disjointness from the new task before running.** Overlap makes
  forgetting unmeasurable; document the check.
- Freeze the item set, the prompt template, and the few-shot examples (including
  their order) at project start.

**Forgetting metric.** Report the raw old-task score and, where a normalised
number is needed:

```
forgetting = score_base - score_checkpoint          # absolute points
relative   = (score_base - score_checkpoint) / (score_base - score_floor)
```

where `score_floor` is chance or an empty-model baseline. Always show absolute
points somewhere — relative metrics hide small denominators.

## New-task performance

Measured with the same rigour, because "forgets less" is only interesting at
comparable new-task gain. Every matched-KL comparison reports both. A method that
forgets less *and* learns less has not been shown to be protective.

## Decoding and normalisation

Identical for every arm and every checkpoint:

- Greedy decoding for scored benchmarks unless the benchmark demands sampling; if
  sampling, fix temperature, top-p and the seed, and average over a fixed number
  of samples.
- Fixed `max_new_tokens`, fixed stop sequences.
- One answer-extraction/normalisation function shared by all arms. Log the parse
  failure rate per checkpoint — a rising failure rate is format drift masquerading
  as forgetting.
- Log completion length distributions next to the scores.

## Output format

One JSON record per (checkpoint, evaluation) appended to
`outputs/results/{run_id}.jsonl`, carrying at minimum:

```
run_id, method, seed, step, git_commit, config_hash,
kl_new, kl_new_se, kl_old, kl_old_se, kl_n_tokens,
new_task_score, old_task_scores{...}, parse_failure_rate,
mean_completion_len, eval_prompt_set_hash, wall_clock
```

Analysis reads only these files. If a number is not in the record, it does not
exist for the paper.

## Sanity checks before trusting a number

- The base model evaluated through this exact path reproduces its known scores.
- `kl_new` and `kl_old` are both ≈ 0 for the base model itself.
- KL is monotone-ish in training step within a run; a non-monotone jump means a
  bad checkpoint or an estimator bug, not an interesting finding.
- Two evaluations of the same checkpoint agree exactly under greedy decoding.
