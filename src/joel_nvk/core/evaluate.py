"""Task evaluation.

One code path per task, run identically for every model state. Alongside accuracy
it reports the parse failure rate and completion length: both are leading
indicators of format drift, which looks like forgetting on a score but is not.
"""

import logging
from collections import defaultdict
from typing import Any

from joel_nvk.core.generate import generate
from joel_nvk.core.scoring import extract_boxed, extract_choice, math_equal
from joel_nvk.data.prompts import Example, render_prompt

logger = logging.getLogger(__name__)


def _completion_stats(records: list[dict[str, Any]]) -> dict[str, float]:
    lengths = [len(record["token_ids"]) for record in records]
    return {
        "mean_completion_len": sum(lengths) / len(lengths) if lengths else 0.0,
        "max_completion_len": float(max(lengths)) if lengths else 0.0,
    }


def evaluate_math(
    model: Any,
    tokenizer: Any,
    examples: list[Example],
    *,
    max_new_tokens: int,
    batch_size: int,
) -> dict[str, Any]:
    """Accuracy on MATH, overall and per difficulty level."""
    prompts = [render_prompt(tokenizer, ex.prompt) for ex in examples]
    completions = generate(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    correct = 0
    unparsed = 0
    by_level: dict[int | None, list[bool]] = defaultdict(list)
    predictions = []

    for example, completion in zip(examples, completions, strict=True):
        gold = extract_boxed(example.target)
        predicted = extract_boxed(completion["text"])
        if predicted is None:
            unparsed += 1
            is_correct = False
        else:
            is_correct = gold is not None and math_equal(predicted, gold)
        correct += int(is_correct)
        by_level[example.meta.get("level")].append(is_correct)
        predictions.append(
            {
                "task": "math",
                "level": example.meta.get("level"),
                "subject": example.meta.get("subject"),
                "gold": gold,
                "predicted": predicted,
                "correct": is_correct,
                "completion": completion["text"],
            }
        )

    total = len(examples)
    return {
        "task": "math",
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "parse_failure_rate": unparsed / total if total else 0.0,
        "accuracy_by_level": {
            str(level): sum(flags) / len(flags)
            for level, flags in sorted(
                by_level.items(), key=lambda item: (item[0] is None, item[0])
            )
        },
        **_completion_stats(completions),
        "predictions": predictions,
    }


def evaluate_mmlu(
    model: Any,
    tokenizer: Any,
    examples: list[Example],
    *,
    max_new_tokens: int,
    batch_size: int,
) -> dict[str, Any]:
    """Accuracy on MMLU, graded on the generated option letter."""
    prompts = [render_prompt(tokenizer, ex.prompt) for ex in examples]
    completions = generate(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    correct = 0
    unparsed = 0
    predictions = []
    for example, completion in zip(examples, completions, strict=True):
        predicted = extract_choice(completion["text"])
        if predicted is None:
            unparsed += 1
            is_correct = False
        else:
            is_correct = predicted == example.target
        correct += int(is_correct)
        predictions.append(
            {
                "task": "mmlu",
                "subject": example.meta.get("subject"),
                "gold": example.target,
                "predicted": predicted,
                "correct": is_correct,
                "completion": completion["text"],
            }
        )

    total = len(examples)
    return {
        "task": "mmlu",
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "parse_failure_rate": unparsed / total if total else 0.0,
        **_completion_stats(completions),
        "predictions": predictions,
    }
