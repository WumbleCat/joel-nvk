"""Task-generic evaluation for the proof-of-concept.

One code path for every task and every checkpoint: render, generate greedily,
grade with the task's own grader, keep every prediction. The per-example
records are what later let us count correct→wrong transitions rather than
trusting an aggregate accuracy.
"""

from __future__ import annotations

import logging
from typing import Any

from joel_nvk.core.generate import generate
from joel_nvk.data.prompts import Example, render_prompt
from joel_nvk.data.tasks import Task

logger = logging.getLogger(__name__)


def evaluate_task(
    model: Any,
    tokenizer: Any,
    task: Task,
    examples: list[Example],
    *,
    batch_size: int,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Accuracy, parse failure rate, completion length and per-example predictions."""
    prompts = [render_prompt(tokenizer, ex.prompt) for ex in examples]
    completions = generate(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens or task.max_new_tokens,
        batch_size=batch_size,
    )

    budget = max_new_tokens or task.max_new_tokens
    predictions = []
    correct = unparsed = bad_format = truncated = 0
    for index, (example, completion) in enumerate(zip(examples, completions, strict=True)):
        is_correct, predicted = task.grade(completion["text"], example)
        format_ok = task.format_ok(completion["text"])
        hit_cap = len(completion["token_ids"]) >= budget
        correct += int(is_correct)
        unparsed += int(predicted is None)
        bad_format += int(not format_ok)
        truncated += int(hit_cap)
        predictions.append(
            {
                "task": task.name,
                "index": index,
                "id": example.meta.get("id"),
                "gold": example.meta.get("answer") or example.target,
                "predicted": predicted,
                "correct": is_correct,
                "format_ok": format_ok,
                "truncated": hit_cap,
                "n_tokens": len(completion["token_ids"]),
                "completion": completion["text"],
            }
        )

    total = len(examples)
    lengths = [p["n_tokens"] for p in predictions]
    result = {
        "task": task.name,
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "parse_failure_rate": unparsed / total if total else 0.0,
        # Answer protocol not followed (no letter / no box) — format drift.
        "format_failure_rate": bad_format / total if total else 0.0,
        # Hit the generation cap — an evaluation-budget problem, not a model one.
        "truncation_rate": truncated / total if total else 0.0,
        "mean_completion_len": sum(lengths) / total if total else 0.0,
        "predictions": predictions,
    }
    logger.info(
        "%s: acc %.3f, parse-fail %.3f, format-fail %.3f, truncated %.3f, mean len %.0f (n=%d)",
        task.name,
        result["accuracy"],
        result["parse_failure_rate"],
        result["format_failure_rate"],
        result["truncation_rate"],
        result["mean_completion_len"],
        total,
    )
    return result


def transitions(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, int]:
    """Count per-example correctness changes between two prediction lists.

    Both lists must come from the same frozen example set in the same order;
    the index is checked so a mismatch fails loudly instead of averaging into
    a plausible-looking number.
    """
    if len(before) != len(after):
        raise ValueError(f"Prediction lists differ in length: {len(before)} vs {len(after)}")
    counts = {"correct_to_wrong": 0, "wrong_to_correct": 0, "stayed_correct": 0, "stayed_wrong": 0}
    for b, a in zip(before, after, strict=True):
        if b["index"] != a["index"]:
            raise ValueError(f"Prediction order mismatch at {b['index']} vs {a['index']}")
        if b["correct"] and not a["correct"]:
            counts["correct_to_wrong"] += 1
        elif not b["correct"] and a["correct"]:
            counts["wrong_to_correct"] += 1
        elif b["correct"]:
            counts["stayed_correct"] += 1
        else:
            counts["stayed_wrong"] += 1
    return counts
