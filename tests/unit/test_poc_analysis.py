from joel_nvk.poc.analysis import checkpoint_rows, flags, pearson, summarise_arm


def _record(arm, step, old, new, kl_old, kl_new, c2w=0):
    return {
        "arm": arm,
        "seed": 1,
        "step": step,
        "old": {
            "task": "arc_challenge",
            "accuracy": old,
            "parse_failure_rate": 0.0,
            "mean_completion_len": 2,
        },
        "new": {
            "task": "gsm8k",
            "accuracy": new,
            "parse_failure_rate": 0.0,
            "mean_completion_len": 100,
        },
        "kl": {
            "kl_old": {"kl": kl_old, "se": 0.01},
            "kl_old_rev": {"kl": kl_old * 1.1},
            "kl_new": {"kl": kl_new, "se": 0.01},
            "kl_new_rev": {"kl": kl_new * 1.1},
        },
        "transitions": {"correct_to_wrong": c2w, "wrong_to_correct": 0},
        "train_loss": 1.0,
        "lr": 1e-4,
    }


RECORDS = [
    _record("sft", 0, 0.60, 0.50, 0.0, 0.0),
    _record("sft", 50, 0.55, 0.60, 0.05, 0.10, c2w=15),
    _record("sft", 100, 0.50, 0.65, 0.10, 0.30, c2w=30),
]


class TestRows:
    def test_forgetting_is_measured_against_step_zero(self):
        rows = checkpoint_rows(RECORDS)
        by_step = {r["step"]: r for r in rows}
        assert by_step[0]["forgetting"] == 0.0
        assert abs(by_step[100]["forgetting"] - 0.10) < 1e-9

    def test_rows_are_sorted_by_step(self):
        assert [r["step"] for r in checkpoint_rows(list(reversed(RECORDS)))] == [0, 50, 100]

    def test_both_kl_axes_survive(self):
        row = checkpoint_rows(RECORDS)[-1]
        assert row["kl_old"] == 0.10
        assert row["kl_new"] == 0.30
        assert row["kl_new_rev"] > row["kl_new"]


class TestSummary:
    def test_headline_numbers(self):
        s = summarise_arm(checkpoint_rows(RECORDS))
        assert s["old_acc_start"] == 0.60
        assert s["old_acc_end"] == 0.50
        assert s["new_acc_max_step"] == 100
        assert abs(s["forgetting_max"] - 0.10) < 1e-9
        assert s["correct_to_wrong_end"] == 30


class TestPearson:
    def test_perfect_correlation(self):
        assert abs(pearson([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-9

    def test_too_few_points_is_none(self):
        assert pearson([1, 2], [2, 4]) is None

    def test_constant_series_is_none(self):
        assert pearson([1, 1, 1], [1, 2, 3]) is None


class TestFlags:
    def test_invalid_stage_b_is_flagged(self):
        m0_m1 = {
            "m0": {"old": {"accuracy": 0.6, "parse_failure_rate": 0}},
            "m1": {"old": {"accuracy": 0.58, "parse_failure_rate": 0}},
        }
        out = flags({}, m0_m1)
        assert any("did not improve Task A" in f for f in out)

    def test_healthy_run_raises_no_flags(self):
        rows = checkpoint_rows(RECORDS)
        m0_m1 = {
            "m0": {"old": {"accuracy": 0.5, "parse_failure_rate": 0}},
            "m1": {"old": {"accuracy": 0.6, "parse_failure_rate": 0}},
        }
        assert flags({"sft": rows}, m0_m1) == []

    def test_no_new_task_learning_is_flagged(self):
        rows = checkpoint_rows(
            [_record("sft", 0, 0.6, 0.5, 0, 0), _record("sft", 10, 0.6, 0.4, 0.1, 0.1)]
        )
        assert any("Task B did not improve" in f for f in flags({"sft": rows}, None))
