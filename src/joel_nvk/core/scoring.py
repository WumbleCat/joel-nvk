"""Answer extraction and grading.

Pure string logic, deliberately shared by every model state: one extraction path
means a score difference between states can never be an extraction difference.
Parse failures are counted, not silently graded as wrong-looking text — a rising
failure rate is format drift, which is a different finding from forgetting.
"""

import re
from fractions import Fraction

BOXED = "\\boxed"

# Ordered from most to least explicit. Only the first is case-insensitive: a bare
# lower-case "a" is far more often the English article than an option letter.
_CHOICE_PATTERNS = (
    re.compile(r"\b(?:answer|option|choice)\b\D{0,20}?([A-H])\b", re.IGNORECASE),
    re.compile(r"^\W{0,4}([A-H])\b"),
    re.compile(r"\b([A-H])\b"),
)

_FRAC_RE = re.compile(r"^\\d?frac\{([^{}]+)\}\{([^{}]+)\}$")
_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def extract_boxed(text: str) -> str | None:
    """Return the contents of the last ``\\boxed{...}``, or None if absent.

    Brace-matched rather than regex-matched, so nested braces (``\\frac{1}{2}``)
    survive intact.
    """
    start = text.rfind(BOXED)
    if start == -1:
        return None
    cursor = start + len(BOXED)
    while cursor < len(text) and text[cursor] != "{":
        if not text[cursor].isspace():
            return None
        cursor += 1
    if cursor >= len(text):
        return None

    depth = 0
    for index in range(cursor, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[cursor + 1 : index]
    return None


def normalize_math_answer(answer: str) -> str:
    """Strip the LaTeX noise that makes equal answers compare unequal."""
    text = answer.strip()
    for token in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "$", "\\$"):
        text = text.replace(token, "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\mbox\{([^{}]*)\}", r"\1", text)
    text = text.replace("^\\circ", "").replace("^{\\circ}", "")
    text = text.replace("\\%", "").replace("%", "")
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".")
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    # A leading "x=" or "answer=" carries no information about correctness.
    text = re.sub(r"^[a-zA-Z]+=", "", text)
    return text


def _as_fraction(text: str) -> Fraction | None:
    """Best-effort numeric reading of a normalised answer."""
    match = _FRAC_RE.match(text)
    if match:
        try:
            return Fraction(int(match.group(1)), int(match.group(2)))
        except (ValueError, ZeroDivisionError):
            return None
    if "/" in text:
        parts = text.split("/")
        if len(parts) == 2 and all(_NUMBER_RE.match(p) for p in parts):
            try:
                return Fraction(parts[0]) / Fraction(parts[1])
            except (ValueError, ZeroDivisionError):
                return None
        return None
    if _NUMBER_RE.match(text):
        try:
            return Fraction(text)
        except ValueError:
            return None
    return None


def math_equal(predicted: str, gold: str) -> bool:
    """Grade a MATH answer: normalised string match, then a numeric fallback."""
    left = normalize_math_answer(predicted)
    right = normalize_math_answer(gold)
    if left == right:
        return True
    left_num, right_num = _as_fraction(left), _as_fraction(right)
    return left_num is not None and left_num == right_num


def extract_choice(text: str) -> str | None:
    """Pull an MMLU option letter out of a completion, or None if there is none."""
    head = text.strip()
    if not head:
        return None
    for pattern in _CHOICE_PATTERNS:
        match = pattern.search(head)
        if match:
            return match.group(1).upper()
    return None
