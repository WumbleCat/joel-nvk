"""Self-SFT data construction: the model's own verified answers as targets.

For every training prompt, sample ``k`` completions from the current policy,
grade them with the task's verifier, and keep correct ones. The result is an
SFT dataset whose targets are on-policy for the model that generated them —
the property the on-policy-vs-off-policy question turns on.

Everything about the filter is reported (attempted, solved, retained) because
"too few survive" is a finding, not something to hide.
"""

from __future__ import annotations

import logging
from typing import Any

from joel_nvk.core.generate import generate
from joel_nvk.data.prompts import Example, render_prompt
from joel_nvk.data.tasks import Task

logger = logging.getLogger(__name__)


def build_self_sft_dataset(
    model: Any,
    tokenizer: Any,
    task: Task,
    examples: list[Example],
    *,
    samples_per_prompt: int,
    temperature: float,
    batch_size: int,
    seed: int,
    keep_per_prompt: int = 1,
) -> tuple[list[Example], dict[str, Any]]:
    """Sample, verify, filter. Returns the kept examples and the retention stats.

    Args:
        keep_per_prompt: How many distinct correct completions to keep per prompt.
            1 keeps the dataset the same shape as the vanilla-SFT one (one
            target per question), which keeps the comparison about *where the
            targets come from* rather than how many there are.
    """
    prompts = [render_prompt(tokenizer, ex.prompt) for ex in examples]
    # Repeat each prompt k times so every sample is an independent draw.
    repeated = [p for p in prompts for _ in range(samples_per_prompt)]
    completions = generate(
        model,
        tokenizer,
        repeated,
        max_new_tokens=task.max_new_tokens,
        batch_size=batch_size,
        do_sample=True,
        temperature=temperature,
        seed=seed,
    )

    kept: list[Example] = []
    n_correct_samples = 0
    n_prompts_solved = 0
    for i, example in enumerate(examples):
        group = completions[i * samples_per_prompt : (i + 1) * samples_per_prompt]
        correct_texts: list[str] = []
        for completion in group:
            is_correct, _ = task.grade(completion["text"], example)
            if is_correct:
                n_correct_samples += 1
                text = completion["text"].strip()
                if text not in correct_texts:
                    correct_texts.append(text)
        if correct_texts:
            n_prompts_solved += 1
            for text in correct_texts[:keep_per_prompt]:
                kept.append(
                    Example(
                        prompt=example.prompt, target=text, meta={**example.meta, "source": "self"}
                    )
                )

    stats = {
        "n_prompts": len(examples),
        "samples_per_prompt": samples_per_prompt,
        "temperature": temperature,
        "n_samples": len(repeated),
        "n_correct_samples": n_correct_samples,
        "sample_accuracy": n_correct_samples / len(repeated) if repeated else 0.0,
        "n_prompts_solved": n_prompts_solved,
        "prompt_solve_rate": n_prompts_solved / len(examples) if examples else 0.0,
        "n_kept": len(kept),
        "keep_per_prompt": keep_per_prompt,
    }
    logger.info(
        "self-SFT filter: %d/%d prompts solved (%.1f%%), %d samples correct of %d, %d kept",
        n_prompts_solved,
        len(examples),
        100 * stats["prompt_solve_rate"],
        n_correct_samples,
        len(repeated),
        len(kept),
    )
    return kept, stats
