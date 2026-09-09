import pytest

from joel_nvk.core.scoring import (
    extract_boxed,
    extract_choice,
    math_equal,
    normalize_math_answer,
)


class TestExtractBoxed:
    def test_simple(self):
        assert extract_boxed(r"so the answer is \boxed{42}.") == "42"

    def test_nested_braces_survive(self):
        assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_takes_the_last_box(self):
        assert extract_boxed(r"\boxed{1} ... actually \boxed{2}") == "2"

    def test_missing_box_is_none(self):
        assert extract_boxed("the answer is 42") is None

    def test_unterminated_box_is_none(self):
        assert extract_boxed(r"\boxed{42") is None


class TestNormalise:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (r"\left(3\right)", "(3)"),
            (r"\dfrac{1}{2}", r"\frac{1}{2}"),
            (r"\text{7}", "7"),
            ("50\\%", "50"),
            ("  12 . ", "12"),
            ("x=5", "5"),
        ],
    )
    def test_strips_latex_noise(self, raw, expected):
        assert normalize_math_answer(raw) == expected


class TestMathEqual:
    def test_exact_match(self):
        assert math_equal("42", "42")

    def test_formatting_difference_is_equal(self):
        assert math_equal(r"\dfrac{1}{2}", r"\frac{1}{2}")

    def test_numeric_fallback(self):
        assert math_equal("1/2", r"\frac{1}{2}")

    def test_different_answers_are_unequal(self):
        assert not math_equal("41", "42")

    def test_zero_denominator_does_not_raise(self):
        assert not math_equal("1/0", "42")


class TestExtractChoice:
    @pytest.mark.parametrize(
        ("completion", "expected"),
        [
            ("B", "B"),
            ("Answer: C", "C"),
            ("(D)", "D"),
            ("**A**", "A"),
            ("The answer is b", "B"),
        ],
    )
    def test_letters(self, completion, expected):
        assert extract_choice(completion) == expected

    def test_no_letter_is_a_parse_failure(self):
        assert extract_choice("I don't know") is None
        assert extract_choice("   ") is None
