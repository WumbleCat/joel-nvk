"""Graders and target construction for the task registry — no network, no torch."""

import pytest

from joel_nvk.core.task_eval import transitions
from joel_nvk.data.prompts import Example
from joel_nvk.data.tasks import TASKS, get_task, gsm8k_target


class TestGsm8kTarget:
    def test_strips_calculator_annotations_and_boxes_the_answer(self):
        solution = "She has 3+4=<<3+4=7>>7 apples.\n#### 7"
        target, answer = gsm8k_target(solution)
        assert "<<" not in target
        assert target.endswith("\\boxed{7}.")
        assert answer == "7"

    def test_removes_thousands_separators(self):
        _, answer = gsm8k_target("Total is 1,200.\n#### 1,200")
        assert answer == "1200"


class TestGrading:
    def test_mc_task_grades_on_the_letter(self):
        task = get_task("arc_challenge")
        example = Example("q", "C", {"task": "arc_challenge"})
        assert task.grade("The answer is C.", example) == (True, "C")
        assert task.grade("B", example) == (False, "B")
        assert task.grade("no idea", example) == (False, None)

    def test_math_task_grades_on_the_boxed_value(self):
        task = get_task("gsm8k")
        example = Example("q", "... \\boxed{42}.", {"task": "gsm8k", "answer": "42"})
        assert task.grade("so \\boxed{42}", example) == (True, "42")
        assert task.grade("so \\boxed{41}", example) == (False, "41")
        assert task.grade("forty-two", example) == (False, None)


class TestRegistry:
    def test_every_task_has_an_eval_split_and_a_kind(self):
        for task in TASKS.values():
            assert task.kind in {"mc", "math"}
            assert task.eval_split

    def test_untrainable_tasks_are_marked(self):
        assert not TASKS["hellaswag"].trainable
        assert TASKS["mmlu"].train_split is None

    def test_unknown_task_raises(self):
        with pytest.raises(KeyError):
            get_task("nope")


class TestTransitions:
    def test_counts_each_cell(self):
        before = [{"index": i, "correct": c} for i, c in enumerate([True, True, False, False])]
        after = [{"index": i, "correct": c} for i, c in enumerate([True, False, True, False])]
        assert transitions(before, after) == {
            "correct_to_wrong": 1,
            "wrong_to_correct": 1,
            "stayed_correct": 1,
            "stayed_wrong": 1,
        }

    def test_order_mismatch_is_an_error(self):
        with pytest.raises(ValueError, match="order"):
            transitions([{"index": 0, "correct": True}], [{"index": 1, "correct": True}])


class TestLastNumberFallback:
    def test_prose_answer_is_graded_but_flagged_as_bad_format(self):
        from joel_nvk.data.tasks import last_number

        task = get_task("gsm8k")
        example = Example("q", "... \\boxed{3160}.", {"task": "gsm8k", "answer": "3160"})
        completion = "Total area = 960 + 1200 + 1000 = 3160 square feet."
        assert task.grade(completion, example) == (True, "3160")
        assert task.format_ok(completion) is False
        assert task.format_ok("so \\boxed{3160}") is True
        assert last_number("costs 1,200.") == "1200"
        assert last_number("no digits here") is None
