---
name: answering-questions
description: How to work when the task is to EXPLAIN rather than change — questions about how the code works, what a run's numbers mean, whether a result supports a claim, or what to do next. Read-only: no edits, no "while I was in there" fixes. Use for "what does X do", "why did Y happen", "what do these results show", "is this finding real"; when the request is to build or fix something, use [[changing-the-code]] instead.
---

# Answering questions

The deliverable is an accurate answer, not a changed file. Reading code, results
and logs is expected; editing them is not.

## The line

| Question | Answer it | Change something |
| --- | --- | --- |
| "What does the KL stage do?" | yes | no |
| "Why is the parse failure rate 100%?" | yes | no |
| "Do these numbers support H1?" | yes | no |
| "Is the loss mask right?" | read it and say | no |
| "Fix the loss mask" | — | yes |

If the answer is "this is broken", **say so and stop**. Describe the fix; do not
apply it. The user decides whether a diagnosis becomes a change. The one
exception is when they ask a question and then explicitly ask you to fix it.

## Answer from the artefact, not from memory

- Code questions: read the file and cite `path:line`. Never describe what the
  code probably does.
- Results questions: read `outputs/results/<run_id>.jsonl`, the manifest, and
  `.predictions.jsonl`. Quote the recorded number, name the run id, and prefer
  `uv run scripts/summarize_run.py <run_id>` over recomputing by hand.
- If a claim rests on something you did not open, say which.

## Reporting numbers honestly

Every accuracy or KL you quote carries its context: which **run**, which
**state** (`base` / `math` / `mmlu`), which **axis** (`kl_old` / `kl_new`), how
many **examples and seeds**, and the **parse failure rate** when it is not near
zero. A number without those is not an answer, it is a rumour.

Then say what it does and does not support:

- One seed, one drift point, tiny eval sets ⇒ *"this shows the machinery runs",*
  never *"this shows on-policy training forgets less".*
- A gap smaller than the noise ⇒ *"no reliable difference"*, with the seed count.
- A KL difference inside the estimator's own standard error ⇒ not a difference.

See [[statistical-analysis]] for which comparisons are admissible and
[[evaluation]] for what each metric actually measures.

## Distinguish the three kinds of "why"

1. **What the code does** — verifiable by reading. State it plainly.
2. **Why it produces this number** — often verifiable from predictions and logs.
   Check before asserting; a truncated completion and a wrong answer look the
   same in an accuracy column.
3. **What it means for the hypotheses** — inference. Label it as such, and give
   the evidence that would change your mind.

## Shape of a good answer

Lead with the answer. Then the evidence, then the caveat. Keep it as short as the
question allows — a one-line question gets a one-line answer, not a report.

Do not narrate the search ("I looked in three files and then..."). Do not restate
the question. Do not append next steps unless they were asked for.

## Unknowns

"I don't know, and here is what would tell us" is a complete answer. Say which
file, record or run would settle it, and offer to go look. Guessing about a
number that exists in a results file is never acceptable — the file is right
there.
