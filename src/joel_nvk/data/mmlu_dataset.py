"""MMLU: the second task, trained on after MATH to induce forgetting.

Framed as generation rather than scoring — the model produces the option letter —
so that training and evaluation use the same interface as MATH.
"""

import logging

from joel_nvk.data.prompts import Example, answer_letter, mmlu_prompt

logger = logging.getLogger(__name__)


def load_mmlu(
    split: str,
    *,
    dataset: str,
    config: str = "all",
    size: int | None = None,
    seed: int = 0,
) -> list[Example]:
    """Load MMLU rows as ``Example``s whose target is the answer letter.

    Args:
        split: e.g. ``"auxiliary_train"`` for training, ``"test"`` for evaluation.
        dataset: HF dataset id (``cais/mmlu``).
        config: Subject config; ``"all"`` spans every subject.
        size: Subsample to this many examples, shuffled with ``seed`` first.
        seed: Shuffle seed — fixes which examples the frozen set contains.
    """
    from datasets import load_dataset

    rows = load_dataset(dataset, config, split=split)
    rows = rows.shuffle(seed=seed)
    if size is not None:
        rows = rows.select(range(min(size, len(rows))))

    examples = []
    for row in rows:
        choices = list(row["choices"])
        letter = answer_letter(int(row["answer"]))
        examples.append(
            Example(
                prompt=mmlu_prompt(row["question"], choices),
                target=letter,
                meta={
                    "task": "mmlu",
                    "subject": row.get("subject"),
                    "n_choices": len(choices),
                },
            )
        )
    logger.info("Loaded %d MMLU examples from %s[%s]", len(examples), dataset, split)
    return examples
