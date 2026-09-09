"""LoRA supervised fine-tuning.

One trainer serves both stages: MATH first, then MMLU continuing from the MATH
adapter. Only the dataset and the hyperparameters in the config change, so the
two stages cannot differ in plumbing.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from joel_nvk.data.collate import encode_dataset, make_collator
from joel_nvk.data.prompts import Example

logger = logging.getLogger(__name__)


def train_lora(
    model: Any,
    tokenizer: Any,
    examples: list[Example],
    *,
    train_cfg: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    run_name: str,
) -> dict[str, Any]:
    """Fine-tune the active LoRA adapter and save it to ``output_dir``.

    Returns the training summary (steps, final loss, dataset size) for the run
    record. The model is left in eval mode.
    """
    import torch
    from transformers import Trainer, TrainingArguments

    encoded = encode_dataset(tokenizer, examples, int(train_cfg["max_length"]))
    if not encoded:
        raise ValueError("Nothing left to train on after encoding")

    device = next(model.parameters()).device
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        overwrite_output_dir=True,
        num_train_epochs=float(train_cfg.get("epochs", 1)),
        max_steps=int(train_cfg.get("max_steps", -1)),
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        gradient_accumulation_steps=int(train_cfg.get("grad_accum", 1)),
        learning_rate=float(train_cfg["lr"]),
        lr_scheduler_type=str(train_cfg.get("scheduler", "cosine")),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_strategy="no",
        report_to=[],
        seed=seed,
        data_seed=seed,
        bf16=use_bf16,
        run_name=run_name,
        disable_tqdm=False,
    )

    model.train()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded,
        data_collator=make_collator(tokenizer),
    )
    result = trainer.train()
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Saved adapter to %s", output_dir)

    return {
        "n_examples": len(encoded),
        "steps": int(result.global_step),
        "train_loss": float(result.training_loss),
        "epochs": float(train_cfg.get("epochs", 1)),
        "lr": float(train_cfg["lr"]),
        "adapter_path": str(output_dir),
    }
