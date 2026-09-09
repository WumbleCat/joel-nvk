---
name: changing-the-code
description: How to work when the task is to MODIFY this repo — write or edit code, configs, adapters, prompts or docs. Covers the gate every change must pass, what invalidates existing results, when to branch, and what to say when done. Use whenever the request asks for something to be built, fixed, added, refactored, renamed, or made to run; not for questions about existing code or results, which belong to [[answering-questions]].
---

# Changing the code

You are modifying a research repo whose output is numbers in a paper. A change
that runs cleanly and quietly makes a number wrong is worse than one that
crashes.

## Before writing anything

Answer these first, in your head or out loud:

1. **Does this invalidate results already collected?** Touching the KL estimator,
   a prompt set, the answer normaliser, the loss mask, the chat template, an eval
   generation setting, or a dataset build changes what previous numbers mean. If
   it does, say so before you write the code, and say which runs must be re-run.
2. **Which side of the boundary does it belong on?** Anything producing a number
   goes in `src/` with a test; scripts stay argument parsing plus a call in;
   notebooks are scratch. See [[repo-standards]].
3. **Does it break symmetry between the arms?** A change applied to one training
   path and not the others manufactures a method effect. See [[training-pipeline]].

## While writing

- Match the surrounding code: same naming, same comment density, same idiom.
  New code should be unidentifiable as new.
- Config, not constants. A run must stay describable by
  `(config, seed, git commit)`.
- Every number-producing function gets a test in the same change, not later.
- Comments explain **why** — especially numerical, statistical and memory
  choices. The what is already in the code.
- Prefer editing the one place a thing is defined over adding a second place.

## The gate before you call it done

```bash
uv run pytest          # must pass
uv run ruff check .    # must be clean
uv run mypy            # must be clean
```

For anything touching the model path, tests and types are not enough — **run it**.
A stage that constructs `TrainingArguments` or calls `generate` can only be
proven by execution, and library majors move under you. `-LowMemory` runs one
stage per process when memory is tight.

## Scope

Do what was asked, completely, and stop. Do not add docs, refactors, formatting
sweeps or extra features that were not requested. If you find a real problem
outside the scope, finish the task and name the problem — do not silently fix it
in the same change.

## Version churn is a live hazard here

transformers, peft, datasets and torch move fast and this repo pins them loosely.
When something fails with `unexpected keyword argument`, `has no attribute`, or a
deprecation notice, the fix is to check the installed version's actual API and
adapt, not to pin backwards. Record the reason in a comment (`# transformers >=5
renamed X to Y`) so the next reader does not undo it.

## Committing

Follow the branch workflow: small self-contained changes go straight to `main`;
anything multi-file gets `feature/<name>`, commits as you go, then a `--no-ff`
merge. Run the gate before each commit so every commit is a usable checkpoint.
New dependencies arrive via `uv add` with `uv.lock` in the same diff.

## What to report when finished

State plainly:

- What changed and where.
- What you actually verified, and how (tests? a real run? nothing?).
- What is **not** verified — untested paths deserve a sentence, not silence.
- Whether previously collected results are still valid.

Never report success for a path you did not execute. "Tests pass and it imports"
is a different claim from "the stage ran", and the difference matters here.
