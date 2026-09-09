"""Model, tokenizer and LoRA adapter loading.

Every arm loads through this module so that base model, tokenizer, dtype and
adaptation surface cannot silently differ between them.

The multi-adapter loader is what makes the KL probe cheap: one set of base
weights in memory, with the base / MATH / MMLU distributions reached by switching
adapters rather than by holding three models at once.
"""

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASE_STATE = "base"


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(requested: str, device: str) -> Any:
    """Pick a compute dtype. ``auto`` means bf16 on GPU, fp32 on CPU."""
    import torch

    if requested != "auto":
        return getattr(torch, requested)
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_tokenizer(model_id: str, *, padding_side: str = "right") -> Any:
    """Load a tokenizer, guaranteeing a pad token.

    Use ``padding_side="left"`` for generation (right padding corrupts the
    continuation) and ``"right"`` for training.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_base_model(model_id: str, *, device: str = "auto", dtype: str = "auto") -> Any:
    """Load the frozen base model in eval mode."""
    from transformers import AutoModelForCausalLM

    device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, device)
    logger.info("Loading %s on %s (%s)", model_id, device, torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return model


def attach_lora(model: Any, lora: Mapping[str, Any], *, adapter_name: str) -> Any:
    """Wrap a base model in a fresh, trainable LoRA adapter."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config, adapter_name=adapter_name)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info("LoRA %s: %d trainable / %d total params", adapter_name, trainable, total)
    return peft_model


def load_adapter_for_training(model: Any, path: Path, *, adapter_name: str) -> Any:
    """Reload a saved adapter and keep training it (stage 2 continues stage 1)."""
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(
        model, str(path), adapter_name=adapter_name, is_trainable=True
    )
    peft_model.set_adapter(adapter_name)
    return peft_model


def load_adapters(model: Any, adapters: Mapping[str, Path]) -> Any:
    """Attach several saved adapters to one base model for evaluation.

    Returns the base model unchanged when ``adapters`` is empty, so callers can
    treat the base-only case uniformly.
    """
    from peft import PeftModel

    items = list(adapters.items())
    if not items:
        return model

    first_name, first_path = items[0]
    peft_model = PeftModel.from_pretrained(model, str(first_path), adapter_name=first_name)
    for name, path in items[1:]:
        peft_model.load_adapter(str(path), adapter_name=name)
    peft_model.eval()
    logger.info("Loaded adapters: %s", ", ".join(name for name, _ in items))
    return peft_model


@contextmanager
def adapter_state(model: Any, state: str) -> Iterator[Any]:
    """Run a block with the model in one of its states.

    ``state == "base"`` disables every adapter, giving exactly the base model's
    distribution; any other value activates that named adapter.
    """
    if not hasattr(model, "set_adapter"):
        if state != BASE_STATE:
            raise ValueError(f"Model has no adapters; cannot select state {state!r}")
        yield model
        return

    if state == BASE_STATE:
        with model.disable_adapter():
            yield model
        return

    previous = getattr(model, "active_adapter", None)
    model.set_adapter(state)
    try:
        yield model
    finally:
        if previous is not None:
            model.set_adapter(previous)
