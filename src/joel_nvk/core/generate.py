"""Batched generation.

Greedy by default: every state is decoded identically, so two evaluations of the
same checkpoint return the same text and a score difference cannot come from
sampling noise.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
    batch_size: int = 8,
    do_sample: bool = False,
    temperature: float = 1.0,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate a completion per prompt.

    Args:
        prompts: Chat-template-rendered prompt strings.
        seed: Only meaningful when ``do_sample`` is true.

    Returns:
        One dict per prompt with the decoded ``text`` and the raw
        ``token_ids`` of the completion — the KL probe reuses the ids so that it
        scores exactly the tokens that were generated.
    """
    import torch

    if tokenizer.padding_side != "left":
        raise ValueError("Generation requires a left-padded tokenizer")
    if do_sample and seed is not None:
        torch.manual_seed(seed)

    device = next(model.parameters()).device
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    results: list[dict[str, Any]] = []
    started = time.monotonic()
    n_batches = -(-len(prompts) // batch_size)
    for batch_index, start in enumerate(range(0, len(prompts), batch_size), start=1):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=None,
                top_k=None,
                pad_token_id=pad_id,
            )

        prompt_width = encoded["input_ids"].shape[1]
        for row in output[:, prompt_width:]:
            ids = [int(token) for token in row]
            if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in ids:
                ids = ids[: ids.index(tokenizer.eos_token_id) + 1]
            results.append(
                {
                    "token_ids": ids,
                    "text": tokenizer.decode(ids, skip_special_tokens=True),
                }
            )
        elapsed = time.monotonic() - started
        tokens = sum(len(r["token_ids"]) for r in results)
        logger.info(
            "generated %d/%d prompts (batch %d/%d, %.0f tokens, %.1f tok/s, %.0fs elapsed)",
            len(results),
            len(prompts),
            batch_index,
            n_batches,
            tokens,
            tokens / elapsed if elapsed else 0.0,
            elapsed,
        )

    return results
