"""Tokenisation and batching for supervised fine-tuning.

The one thing that must never drift between arms is the loss mask: the prompt is
always masked, the target never is. An arm that trains on its own prompt tokens
would look different for reasons that have nothing to do with the method.
"""

import logging
from typing import Any

from joel_nvk.data.prompts import Example, render_prompt

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def encode_example(
    tokenizer: Any,
    example: Example,
    max_length: int,
) -> dict[str, list[int]] | None:
    """Tokenise one example into ``input_ids``/``labels`` with the prompt masked.

    Returns ``None`` when the prompt alone does not fit in ``max_length`` — such
    an example is dropped rather than silently truncated into nonsense.
    """
    prompt_text = render_prompt(tokenizer, example.prompt)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(example.target, add_special_tokens=False)["input_ids"]

    if tokenizer.eos_token_id is not None:
        target_ids = [*target_ids, tokenizer.eos_token_id]

    if len(prompt_ids) >= max_length:
        return None

    target_ids = target_ids[: max_length - len(prompt_ids)]
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + target_ids,
    }


def encode_dataset(
    tokenizer: Any,
    examples: list[Example],
    max_length: int,
) -> list[dict[str, list[int]]]:
    """Encode a list of examples, reporting how many were dropped as too long."""
    encoded = [encode_example(tokenizer, ex, max_length) for ex in examples]
    kept = [item for item in encoded if item is not None]
    dropped = len(encoded) - len(kept)
    if dropped:
        logger.warning(
            "Dropped %d/%d examples whose prompt exceeded max_length=%d",
            dropped,
            len(encoded),
            max_length,
        )
    return kept


def make_collator(tokenizer: Any):
    """Build a Trainer-compatible collator that right-pads a batch.

    Labels are padded with ``IGNORE_INDEX`` so padding never contributes loss.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for feature in features:
            ids = feature["input_ids"]
            pad = width - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(feature["labels"] + [IGNORE_INDEX] * pad)
            attention.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    return collate
