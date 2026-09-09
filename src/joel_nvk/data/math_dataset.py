"""MATH: the task whose accuracy is watched for forgetting.

Training uses the train split (all levels by default); evaluation uses a frozen
sample of the test split restricted to levels 3-5.
"""

import logging
import re
from typing import Any

from joel_nvk.data.prompts import Example, math_prompt

logger = logging.getLogger(__name__)

LEVEL_RE = re.compile(r"(\d+)")

DEFAULT_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def parse_level(raw: Any) -> int | None:
    """MATH stores levels as ``"Level 3"``; a few rows are ``"Level ?"``."""
    if raw is None:
        return None
    match = LEVEL_RE.search(str(raw))
    return int(match.group(1)) if match else None


def load_math(
    split: str,
    *,
    dataset: str,
    subjects: tuple[str, ...] | list[str] = DEFAULT_SUBJECTS,
    levels: tuple[int, ...] | list[int] | None = None,
    size: int | None = None,
    seed: int = 0,
) -> list[Example]:
    """Load MATH rows as ``Example``s, optionally filtered by level and subsampled.

    Args:
        split: ``"train"`` or ``"test"``.
        dataset: HF dataset id; the subject configs are loaded and concatenated.
        subjects: Which subject configs to include.
        levels: Keep only these difficulty levels (e.g. ``[3, 4, 5]``).
        size: Subsample to this many examples, shuffled with ``seed`` first.
        seed: Shuffle seed — fixes which examples the frozen set contains.
    """
    from datasets import concatenate_datasets, load_dataset

    parts = []
    for subject in subjects:
        parts.append(load_dataset(dataset, subject, split=split))
    rows = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    if levels is not None:
        wanted = set(levels)
        rows = rows.filter(lambda row: parse_level(row.get("level")) in wanted)

    rows = rows.shuffle(seed=seed)
    if size is not None:
        rows = rows.select(range(min(size, len(rows))))

    examples = [
        Example(
            prompt=math_prompt(row["problem"]),
            target=row["solution"].strip(),
            meta={
                "task": "math",
                "level": parse_level(row.get("level")),
                "subject": row.get("type"),
                "problem": row["problem"],
            },
        )
        for row in rows
    ]
    logger.info("Loaded %d MATH examples from %s[%s]", len(examples), dataset, split)
    return examples


def assert_disjoint(train: list[Example], evaluation: list[Example]) -> None:
    """Fail loudly if any evaluation problem also appears in training.

    Forgetting is unmeasurable on contaminated items, so this runs before every
    pipeline, not only when something looks wrong.
    """
    train_problems = {ex.meta.get("problem", ex.prompt).strip() for ex in train}
    overlap = [
        ex for ex in evaluation if ex.meta.get("problem", ex.prompt).strip() in train_problems
    ]
    if overlap:
        raise ValueError(
            f"{len(overlap)} evaluation problems also appear in the training set — "
            "the forgetting measurement would be contaminated"
        )
