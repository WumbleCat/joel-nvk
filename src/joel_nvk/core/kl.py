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


def _batch_logprobs(model: Any, input_ids: Any, attention_mask: Any, gather_at: Any) -> Any:
    """Log-softmax over the vocabulary at the scored positions only.

    KL is only ever read at continuation positions, and the prompt is usually the
    long part of the sequence — a MATH prompt runs several hundred tokens against
    a continuation of a few dozen. Slicing before the softmax keeps the resident
    tensor proportional to the continuation, not the whole sequence, which is the
    difference between a few hundred MB and a few GB per state.
    """
    import torch

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Position j-1 predicts token j, so the scored logits sit one step earlier.
    index = gather_at.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
    scored = torch.gather(logits, 1, index)
    return torch.log_softmax(scored.float(), dim=-1)


def _pair_kl(logp_policy: Any, logp_reference: Any, valid: Any) -> Any:
    """Per-sequence mean KL(policy || reference) over the valid scored positions."""
    pointwise = logp_policy.exp() * (logp_policy - logp_reference)
    per_token = pointwise.sum(dim=-1) * valid
    counts = valid.sum(dim=-1).clamp(min=1)
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

        longest_continuation = max(len(item["continuation_ids"]) for item in batch)

        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        # For each row, the logit positions that predict its continuation tokens.
        gather_at = torch.zeros((len(batch), longest_continuation), dtype=torch.long)
        valid = torch.zeros((len(batch), longest_continuation), dtype=torch.float)

        for row, (item, seq) in enumerate(zip(batch, sequences, strict=True)):
            n_prompt = len(item["prompt_ids"])
            n_cont = len(item["continuation_ids"])
            input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention[row, : len(seq)] = 1
            gather_at[row, :n_cont] = torch.arange(n_prompt - 1, n_prompt + n_cont - 1)
            valid[row, :n_cont] = 1.0

        input_ids = input_ids.to(device)
        attention = attention.to(device)
        gather_at = gather_at.to(device)
        valid = valid.to(device)
        total_tokens += int(valid.sum().item())

        logprobs = {}
        for state in states:
            with adapter_state(model, state):
                logprobs[state] = _batch_logprobs(model, input_ids, attention, gather_at)

        for policy, reference in pairs:
            values = _pair_kl(logprobs[policy], logprobs[reference], valid)
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
