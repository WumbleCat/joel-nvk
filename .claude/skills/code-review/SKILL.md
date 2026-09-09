---
name: code-review
description: Review checklist for this repo's research code, where the failure mode that matters is a silently wrong number rather than a crash — KL correctness, arm asymmetries, eval leakage, analysis errors, and provenance. Use when reviewing a diff, before merging training/eval/analysis changes, and before any run whose results are intended for the paper.
---

# Code review

Ordinary review catches crashes. Here the dangerous bug is code that runs
cleanly and produces a number that is wrong in a direction that flatters the
hypothesis. Review in that order of priority.

## 1. Does this change invalidate existing numbers?

Ask first, every time. A change to the KL estimator, a prompt set, the answer
normaliser, the loss mask, the chat template, or the eval batch size invalidates
previously collected results on that axis.

If it does: the diff must say so, and the affected results must be re-run or
quarantined. Mixing estimator versions in one figure is the worst outcome this
repo can produce.

## 2. Arm asymmetry

The central threat to validity: something differs between `sft`, `self_sft`,
`iter_sft` and `grpo` that is not the method.

- Same base model, tokenizer, precision, adaptation surface (LoRA rank / target
  modules), max length, truncation side.
- Same prompt formatting and chat template, and the **same prompt-token loss
  masking**. An unmasked prompt in one arm alone manufactures a method effect.
- Shared trainer plumbing actually shared, not copy-pasted with a drifted default.
- Any deliberate difference comes from config and is recorded in the manifest.

## 3. KL correctness

- Direction is consistent everywhere: KL(policy ‖ base) or KL(base ‖ policy),
  fixed once, named in the code. A silently flipped direction changes the ordering
  of arms.
- Averaged over tokens with the token count recorded; padding and special tokens
  excluded from the average.
- Computed in float32; log-probs from log-softmax, never `log(softmax(x))`.
- The base model is genuinely frozen and in eval mode — no dropout, no LoRA
  adapters still attached, no gradient path.
- `kl_new` and `kl_old` use their own prompt sets and are never assigned to the
  same variable or column.
- Base model against itself gives ≈ 0 on both axes; assert it in a test.

## 4. Evaluation integrity

- No new-task data in the old-task suite; the disjointness check exists and runs.
- Eval prompt sets and few-shot examples are frozen and hashed; nothing in the
  diff regenerates them as a side effect.
- Decoding settings identical across arms and checkpoints.
- One shared answer extraction/normalisation path; parse failure rate logged.
- Evaluation never sees optimiser state, training labels, or the checkpoint's own
  training split.
- Eval is deterministic: same checkpoint twice, same numbers.

## 5. Checkpoint matching

- Matching selects on **achieved** KL, and the achieved value is stored, not the
  target.
- Ties and out-of-range cases are handled explicitly, not by silently taking the
  last checkpoint.
- No selection on outcome anywhere — the metric being studied must not appear in
  the checkpoint-selection code path.
- Runs that fail to reach a rung are recorded as missing, not filled in from a
  neighbour.

## 6. Analysis code

- Bootstrap resamples **seeds**, not benchmark items.
- Comparisons stay inside the KL range all arms cover; no extrapolation.
- Method effects reported before and after conditioning on KL.
- Dropped runs are counted and reported; no silent `dropna()` on the row that
  matters.
- Figures come from committed results files via committed code — no numbers typed
  into a notebook.

## 7. Provenance and hygiene

- Manifest written before training starts; `git_dirty` checked and warned on.
- New dependencies arrive via `uv add` with `uv.lock` updated in the same diff —
  never a manual install that only exists on the author's machine.
- Generation RNG seeded separately from the training RNG.
- Paths go through `joel_nvk.utils.paths`; no hardcoded relative paths.
- Nothing under `outputs/` or `data/` is committed; nothing in `src/` writes
  outside `outputs/`.
- New logic that produces a paper number lives in `src/` and has a test.

## Tests worth demanding in the diff

- KL of the base model against itself is ≈ 0.
- KL of a deliberately perturbed model is > 0 and increases with perturbation size.
- Matching picks the nearest checkpoint on a synthetic KL trajectory, including
  ties and out-of-range targets.
- The forgetting metric is 0 when scores equal the base scores.
- Config merge: `dev.yaml` overrides only the keys it names.

## How to report

Lead with anything in categories 1–3 — those change conclusions. Separate
"this makes a number wrong" from "this is untidy", and say plainly which findings
block a paper run versus which can wait.
