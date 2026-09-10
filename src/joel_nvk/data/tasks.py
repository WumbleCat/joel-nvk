"""Task registry for the proof-of-concept.

A task is a dataset plus the three things the pipeline needs to treat it
uniformly: how to load a split as ``Example``s, how to grade a completion, and
how long a completion is allowed to be. Every task goes through the same
prompt templates and the same graders, so a difference between tasks is a
difference in *data*, never in plumbing.

Two families:

* ``mc``   multiple choice — the model answers with an option letter.
* ``math`` free-form reasoning — the model ends with ``\\boxed{answer}``.

The registry deliberately includes tasks that were probed and rejected (see
POC_DESIGN.md) so the probe can report them next to the chosen pair.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from joel_nvk.core.scoring import extract_boxed, extract_choice, math_equal
from joel_nvk.data.prompts import CHOICE_LETTERS, Example, math_prompt, mmlu_prompt

logger = logging.getLogger(__name__)

GSM8K_CALC_RE = re.compile(r"<<[^>]*>>")


@dataclass(frozen=True)
class Task:
    """One benchmark, ready to be loaded, prompted and graded."""

    name: str
    capability: str
    kind: str  # "mc" | "math"
    dataset: str
    config: str | None
    train_split: str | None  # None: no usable training split
    eval_split: str
    max_new_tokens: int
    loader: Callable[[str, int | None, int], list[Example]]
    trainable: bool = True
    note: str = ""

    def load(self, split: str, *, size: int | None = None, seed: int = 0) -> list[Example]:
        examples = self.loader(split, size, seed)
        logger.info("%s[%s]: %d examples", self.name, split, len(examples))
        return examples

    def grade(self, completion: str, example: Example) -> tuple[bool, str | None]:
        """Return ``(is_correct, predicted)``; ``predicted`` is None on a parse failure.

        Math tasks fall back to the last number in the completion when no
        ``\\boxed{}`` is present (the convention lm-eval uses for GSM8K), so a
        correct answer written in prose still counts. Whether the requested
        format was followed is reported separately by ``format_ok`` — the two
        must not be conflated, because format loss and capability loss are
        different findings.
        """
        if self.kind == "mc":
            predicted = extract_choice(completion)
            return (predicted == example.target if predicted else False), predicted
        predicted = extract_boxed(completion)
        if predicted is None:
            predicted = last_number(completion)
        gold = example.meta.get("answer") or extract_boxed(example.target)
        if predicted is None or gold is None:
            return False, predicted
        return math_equal(predicted, gold), predicted

    def format_ok(self, completion: str) -> bool:
        """Did the completion follow the answer protocol the prompt asked for?"""
        if self.kind == "mc":
            return extract_choice(completion) is not None
        return extract_boxed(completion) is not None


NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text: str) -> str | None:
    """The last number in a completion, thousands separators removed."""
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "").rstrip(".")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _rows(dataset: str, config: str | None, split: str, size: int | None, seed: int) -> Any:
    from datasets import load_dataset

    rows = (
        load_dataset(dataset, config, split=split) if config else load_dataset(dataset, split=split)
    )
    rows = rows.shuffle(seed=seed)
    if size is not None:
        rows = rows.select(range(min(size, len(rows))))
    return rows


def _mc_example(
    task: str, question: str, choices: list[str], gold_index: int, **meta: Any
) -> Example:
    return Example(
        prompt=mmlu_prompt(question, choices),
        target=CHOICE_LETTERS[gold_index],
        meta={"task": task, "n_choices": len(choices), "question": question, **meta},
    )


def _letter_index(label: str, labels: list[str]) -> int:
    """ARC labels are usually A-D but a few items use 1-4; map by position."""
    return labels.index(label)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def _load_arc(config: str) -> Callable[[str, int | None, int], list[Example]]:
    name = "arc_challenge" if config == "ARC-Challenge" else "arc_easy"

    def load(split: str, size: int | None, seed: int) -> list[Example]:
        out = []
        for row in _rows("allenai/ai2_arc", config, split, size, seed):
            labels = list(row["choices"]["label"])
            texts = list(row["choices"]["text"])
            gold = _letter_index(row["answerKey"], labels)
            out.append(_mc_example(name, row["question"], texts, gold, id=row["id"]))
        return out

    return load


def _load_commonsense_qa(split: str, size: int | None, seed: int) -> list[Example]:
    out = []
    for row in _rows("tau/commonsense_qa", None, split, size, seed):
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        gold = labels.index(row["answerKey"])
        out.append(_mc_example("commonsense_qa", row["question"], texts, gold, id=row["id"]))
    return out


def _load_winogrande(split: str, size: int | None, seed: int) -> list[Example]:
    out = []
    for row in _rows("allenai/winogrande", "winogrande_xl", split, size, seed):
        question = (
            "Fill in the blank (_) in the sentence with the option that makes the most sense.\n\n"
            f"{row['sentence']}"
        )
        gold = int(row["answer"]) - 1
        out.append(_mc_example("winogrande", question, [row["option1"], row["option2"]], gold))
    return out


def _load_hellaswag(split: str, size: int | None, seed: int) -> list[Example]:
    out = []
    for row in _rows("Rowan/hellaswag", None, split, size, seed):
        question = f"Which ending most plausibly continues the text?\n\n{row['ctx']}"
        out.append(_mc_example("hellaswag", question, list(row["endings"]), int(row["label"])))
    return out


def _load_mmlu(split: str, size: int | None, seed: int) -> list[Example]:
    out = []
    for row in _rows("cais/mmlu", "all", split, size, seed):
        out.append(
            _mc_example(
                "mmlu",
                row["question"],
                list(row["choices"]),
                int(row["answer"]),
                subject=row.get("subject"),
            )
        )
    return out


def gsm8k_target(solution: str) -> tuple[str, str]:
    """Rewrite a GSM8K solution into the boxed format every math task uses.

    Strips the ``<<3+4=7>>`` calculator annotations and turns the trailing
    ``#### 7`` marker into a boxed final answer, so training targets and the
    grader agree on one convention.
    """
    body, _, answer = solution.rpartition("####")
    answer = answer.strip().replace(",", "")
    body = GSM8K_CALC_RE.sub("", body).strip()
    return f"{body}\nThe final answer is \\boxed{{{answer}}}.", answer


def _load_gsm8k(split: str, size: int | None, seed: int) -> list[Example]:
    out = []
    for row in _rows("openai/gsm8k", "main", split, size, seed):
        target, answer = gsm8k_target(row["answer"])
        out.append(
            Example(
                prompt=math_prompt(row["question"]),
                target=target,
                meta={"task": "gsm8k", "answer": answer, "question": row["question"]},
            )
        )
    return out


def _load_math(split: str, size: int | None, seed: int) -> list[Example]:
    from joel_nvk.data.math_dataset import load_math

    return load_math(split, dataset="EleutherAI/hendrycks_math", size=size, seed=seed)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

TASKS: dict[str, Task] = {
    "arc_challenge": Task(
        name="arc_challenge",
        capability="grade-school science knowledge (MC)",
        kind="mc",
        dataset="allenai/ai2_arc",
        config="ARC-Challenge",
        train_split="train",
        eval_split="test",
        max_new_tokens=8,
        loader=_load_arc("ARC-Challenge"),
        note="train 1,119 / validation 299 / test 1,172; CC-BY-SA 4.0",
    ),
    "arc_easy": Task(
        name="arc_easy",
        capability="grade-school science knowledge, easier items (MC)",
        kind="mc",
        dataset="allenai/ai2_arc",
        config="ARC-Easy",
        train_split="train",
        eval_split="test",
        max_new_tokens=8,
        loader=_load_arc("ARC-Easy"),
        note="train 2,251 / test 2,376; same source as ARC-Challenge; augments its training set",
    ),
    "commonsense_qa": Task(
        name="commonsense_qa",
        capability="commonsense reasoning over ConceptNet relations (5-way MC)",
        kind="mc",
        dataset="tau/commonsense_qa",
        config=None,
        train_split="train",
        eval_split="validation",
        max_new_tokens=8,
        loader=_load_commonsense_qa,
        note="train 9,741 / validation 1,221; test labels hidden; MIT",
    ),
    "winogrande": Task(
        name="winogrande",
        capability="pronoun-resolution commonsense (binary)",
        kind="mc",
        dataset="allenai/winogrande",
        config="winogrande_xl",
        train_split="train",
        eval_split="validation",
        max_new_tokens=8,
        loader=_load_winogrande,
        note="train 40,398 / validation 1,267; binary, so chance is 50% and variance is high",
    ),
    "hellaswag": Task(
        name="hellaswag",
        capability="commonsense sentence completion (4-way MC)",
        kind="mc",
        dataset="Rowan/hellaswag",
        config=None,
        train_split="train",
        eval_split="validation",
        max_new_tokens=8,
        loader=_load_hellaswag,
        trainable=False,
        note="probe only: Chizhov et al. 2025 find ~40% ungrammatical prompts and 65% of "
        "predictions unchanged without the question — rejected for evaluation",
    ),
    "mmlu": Task(
        name="mmlu",
        capability="academic knowledge across 57 subjects (MC)",
        kind="mc",
        dataset="cais/mmlu",
        config="all",
        train_split=None,
        eval_split="test",
        max_new_tokens=8,
        loader=_load_mmlu,
        trainable=False,
        note="control benchmark only: no native train split (auxiliary_train is ARC/RACE/OBQA), "
        "~6.5% label errors per MMLU-Redux",
    ),
    "gsm8k": Task(
        name="gsm8k",
        capability="grade-school arithmetic word problems (free-form CoT)",
        kind="math",
        dataset="openai/gsm8k",
        config="main",
        train_split="train",
        eval_split="test",
        # The probe showed 42/100 M0 solutions truncated at 320 tokens: the
        # instruct model writes long LaTeX-heavy chains of thought.
        max_new_tokens=512,
        loader=_load_gsm8k,
        note="train 7,473 / test 1,319; MIT; ~1.7% label noise; GSM1k finds mild overfitting "
        "in some model families",
    ),
    "math": Task(
        name="math",
        capability="competition mathematics, levels 1-5 (free-form CoT)",
        kind="math",
        dataset="EleutherAI/hendrycks_math",
        config=None,
        train_split="train",
        eval_split="test",
        max_new_tokens=512,
        loader=_load_math,
        note="train 7,500 / test 5,000 across 7 subjects; MIT",
    ),
}


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown task {name!r}; known: {sorted(TASKS)}") from exc
