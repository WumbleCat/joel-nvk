"""KL divergence between model states on a frozen probe set.

Estimator, fixed once and used everywhere:

* **Exact token-level KL**, summed over the full vocabulary at each position —
  not a sampled estimate, so it is reproducible to the last decimal.
* **Teacher-forced** on a frozen set of (prompt, continuation) pairs generated
  once by the base model, so all states are scored on identical tokens.
* **Direction**: ``KL(policy || reference)``, i.e. the expectation is taken under
  the policy. Flipping this reorders the arms, so it is fixed here and named in
  every result field.
* Averaged over continuation tokens (padding and prompt excluded), in float32
  regardless of the model's compute dtype.

Two axes are probed separately and never mixed: ``kl_old`` on MATH prompts (where
forgetting is measured) and ``kl_new`` on MMLU prompts (where training happened).
"""

import logging
from typing import Any

from joel_nvk.models.loading import adapter_state

logger = logging.getLogger(__name__)


def _batch_logprobs(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    """Log-softmax over the vocabulary, in float32, for the given batch."""
    import torch

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return torch.log_softmax(logits[:, :-1, :].float(), dim=-1)


def _pair_kl(logp_policy: Any, logp_reference: Any, selected: Any) -> Any:
    """Per-sequence mean KL(policy || reference) over the selected positions."""
    import torch

    pointwise = torch.exp(logp_policy) * (logp_policy - logp_reference)
    per_token = pointwise.sum(dim=-1)  # [B, T-1]
    per_token = per_token * selected
    counts = selected.sum(dim=-1).clamp(min=1)
    return per_token.sum(dim=-1) / counts


def kl_between_states(
    model: Any,
    tokenizer: Any,
    probe: list[dict[str, Any]],
    pairs: list[tuple[str, str]],
    *,
    batch_size: int = 1,
) -> dict[str, dict[str, float]]:
    """Compute mean KL for each (policy, reference) state pair over the probe.

    Args:
        model: A model carrying the named adapters (see ``models.loading``).
        probe: Items with ``prompt_ids`` and ``continuation_ids``.
        pairs: ``(policy, reference)`` state names, e.g. ``("mmlu", "base")``.
        batch_size: Kept at 1 by default — full-vocabulary log-probs for three
            states at once dominate memory.

    Returns:
        ``{"mmlu||base": {"kl": ..., "se": ..., "n_prompts": ..., "n_tokens": ...}}``
    """
    import torch

    states = sorted({name for pair in pairs for name in pair})
    per_prompt: dict[tuple[str, str], list[float]] = {pair: [] for pair in pairs}
    total_tokens = 0

    device = next(model.parameters()).device
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    for start in range(0, len(probe), batch_size):
        batch = probe[start : start + batch_size]
        sequences = [item["prompt_ids"] + item["continuation_ids"] for item in batch]
        width = max(len(seq) for seq in sequences)

        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        # True at positions holding a continuation token.
        is_continuation = torch.zeros((len(batch), width), dtype=torch.bool)
        for row, (item, seq) in enumerate(zip(batch, sequences, strict=True)):
            input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention[row, : len(seq)] = 1
            is_continuation[row, len(item["prompt_ids"]) : len(seq)] = True

        input_ids = input_ids.to(device)
        attention = attention.to(device)
        # Position j-1 predicts token j, so drop the first column to align.
        selected = is_continuation[:, 1:].to(device).float()
        total_tokens += int(selected.sum().item())

        logprobs = {}
        for state in states:
            with adapter_state(model, state):
                logprobs[state] = _batch_logprobs(model, input_ids, attention)

        for policy, reference in pairs:
            values = _pair_kl(logprobs[policy], logprobs[reference], selected)
            per_prompt[(policy, reference)].extend(float(v) for v in values.cpu())

        del logprobs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        f"{policy}||{reference}": _summarise(values, total_tokens)
        for (policy, reference), values in per_prompt.items()
    }


def _summarise(values: list[float], n_tokens: int) -> dict[str, float]:
    """Mean KL plus a bootstrap standard error over prompts.

    A match tolerance narrower than this SE would be a fiction, so the SE travels
    with every KL number.
    """
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"kl": float("nan"), "se": float("nan"), "n_prompts": 0, "n_tokens": 0}

    rng = np.random.default_rng(0)
    draws = rng.integers(0, array.size, size=(1000, array.size))
    boot = array[draws].mean(axis=1)
    return {
        "kl": float(array.mean()),
        "se": float(boot.std(ddof=1)),
        "n_prompts": int(array.size),
        "n_tokens": int(n_tokens),
    }


def build_probe(
    tokenizer: Any,
    rendered_prompts: list[str],
    completions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Freeze (prompt, continuation) pairs for scoring under every state."""
    probe = []
    for prompt, completion in zip(rendered_prompts, completions, strict=True):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        continuation = completion["token_ids"]
        if not continuation:
            continue
        probe.append({"prompt_ids": prompt_ids, "continuation_ids": continuation})
    logger.info(
        "Probe: %d of %d prompts produced a scorable continuation",
        len(probe),
        len(rendered_prompts),
    )
    return probe
