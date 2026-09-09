import pytest

from joel_nvk.core.pipeline import STAGES, dataset_hash, make_run_id, run_pipeline
from joel_nvk.core.report import format_summary, summarise
from joel_nvk.data.math_dataset import assert_disjoint, parse_level
from joel_nvk.data.prompts import Example
from joel_nvk.utils.seeding import derive_seed


class TestDatasetHash:
    def test_is_stable(self):
        examples = [Example("p", "t")]
        assert dataset_hash(examples) == dataset_hash([Example("p", "t")])

    def test_changes_with_content(self):
        assert dataset_hash([Example("p", "t")]) != dataset_hash([Example("p", "u")])

    def test_field_boundaries_cannot_be_confused(self):
        # "ab"+"c" and "a"+"bc" must not collide.
        assert dataset_hash([Example("ab", "c")]) != dataset_hash([Example("a", "bc")])


class TestLevels:
    @pytest.mark.parametrize(
        ("raw", "expected"), [("Level 3", 3), ("Level 5", 5), ("Level ?", None), (None, None)]
    )
    def test_parse_level(self, raw, expected):
        assert parse_level(raw) == expected


class TestDisjointness:
    def test_overlap_raises(self):
        train = [Example("prompt", "solution", {"problem": "1+1"})]
        evaluation = [Example("prompt", "solution", {"problem": "1+1"})]
        with pytest.raises(ValueError, match="contaminated"):
            assert_disjoint(train, evaluation)

    def test_disjoint_sets_pass(self):
        train = [Example("a", "b", {"problem": "1+1"})]
        evaluation = [Example("c", "d", {"problem": "2+2"})]
        assert_disjoint(train, evaluation)


class TestStageSelection:
    def test_unknown_stage_is_rejected_before_any_work(self):
        config = {"project": {"seed": 1, "name": "t"}, "model": {"id": "x/y"}}
        with pytest.raises(ValueError, match="Unknown stages"):
            run_pipeline(config, stages=["not_a_stage"])

    def test_stage_order_is_fixed(self):
        assert STAGES.index("train_math") < STAGES.index("eval_math")
        assert STAGES.index("eval_math") < STAGES.index("train_mmlu")
        assert STAGES.index("train_mmlu") < STAGES.index("eval_mmlu")

    def test_run_id_carries_model_and_seed(self):
        run_id = make_run_id({"model": {"id": "Qwen/Qwen2.5-0.5B-Instruct"}}, seed=7)
        assert run_id.startswith("pipeline-Qwen2.5-0.5B-Instruct-s7-")


class TestDerivedSeeds:
    def test_labels_give_different_streams(self):
        assert derive_seed(1, "math_eval") != derive_seed(1, "mmlu_eval")

    def test_stable_across_calls(self):
        assert derive_seed(1, "math_eval") == derive_seed(1, "math_eval")


def _evaluation(state, math_acc, mmlu_acc):
    return {
        "stage": f"eval_{state}",
        "state": state,
        "math": {
            "accuracy": math_acc,
            "parse_failure_rate": 0.0,
            "mean_completion_len": 120,
            "accuracy_by_level": {"3": math_acc, "4": math_acc, "5": math_acc},
        },
        "mmlu": {"accuracy": mmlu_acc, "parse_failure_rate": 0.0, "mean_completion_len": 2},
    }


class TestReport:
    def test_summary_reports_the_forgetting_delta(self):
        records = [
            _evaluation("base", 0.20, 0.30),
            _evaluation("math", 0.40, 0.30),
            _evaluation("mmlu", 0.25, 0.55),
            {
                "stage": "kl",
                "pairs": ["math||base"],
                "kl_old": {"math||base": {"kl": 0.1, "se": 0.01}},
                "kl_new": {"math||base": {"kl": 0.2, "se": 0.02}},
            },
        ]
        text = format_summary("run-1", summarise(records))

        assert "change -15.0 points" in text  # 40% -> 25% on MATH
        assert "change +25.0 points" in text  # 30% -> 55% on MMLU
        assert "0.1000 +/- 0.0100" in text

    def test_partial_run_still_renders(self):
        text = format_summary("run-2", summarise([_evaluation("base", 0.2, 0.3)]))
        assert "base" in text
        assert "second training stage" not in text
