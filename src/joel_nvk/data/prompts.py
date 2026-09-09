"""Prompt construction for the two tasks.

Pure string work: no tokenizer, no torch. The tokenizer only enters at
``render_prompt``, which applies the model's chat template.

Both tasks share one instruction style so that a difference between arms cannot
come from a difference in formatting.
"""

from dataclasses import dataclass, field
from typing import Any

MATH_INSTRUCTION = (
    "Solve the following problem. Reason step by step, then give the final answer inside \\boxed{}."
)

MMLU_INSTRUCTION = (
    "Answer the following multiple-choice question. Reply with the single letter of "
    "the correct option and nothing else."
)

CHOICE_LETTERS = ("A", "B", "C", "D", "E", "F", "G", "H")


@dataclass(frozen=True)
class Example:
    """One prompt/target pair plus whatever metadata evaluation needs.

    ``prompt`` is the user-visible message *before* the chat template is applied;
    ``target`` is the text the model should produce.
    """

    prompt: str
    target: str
    meta: dict[str, Any] = field(default_factory=dict)


def math_prompt(problem: str) -> str:
    return f"{MATH_INSTRUCTION}\n\nProblem:\n{problem.strip()}"


def mmlu_prompt(question: str, choices: list[str]) -> str:
    if len(choices) > len(CHOICE_LETTERS):
        raise ValueError(f"Too many choices ({len(choices)}) for the letter set")
    rendered = "\n".join(
        f"{letter}. {choice.strip()}"
        for letter, choice in zip(CHOICE_LETTERS, choices, strict=False)
    )
    return f"{MMLU_INSTRUCTION}\n\nQuestion:\n{question.strip()}\n\nOptions:\n{rendered}"


def answer_letter(index: int) -> str:
    """Map MMLU's integer answer to its option letter."""
    try:
        return CHOICE_LETTERS[index]
    except IndexError as exc:
        raise ValueError(f"Answer index {index} outside the letter set") from exc


def render_prompt(tokenizer: Any, prompt: str) -> str:
    """Apply the model's chat template, leaving the model ready to answer.

    Falls back to a bare instruct format for base models that ship no template,
    so that a missing template is a visible formatting choice rather than a crash.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{prompt}\n\nAnswer:"
