"""Resuming a crashed training stage from its latest adapter snapshot."""

from joel_nvk.models.train import planned_steps
from joel_nvk.poc.pipeline import _latest_snapshot, _resume_plan

CFG = {"batch_size": 4, "grad_accum": 4, "epochs": 2, "max_steps": -1, "warmup_ratio": 0.03}


def _snapshot(root, step, nested=False):
    d = root / f"step-{step}"
    (d / "x").mkdir(parents=True) if nested else d.mkdir(parents=True)
    ((d / "x") if nested else d).joinpath("adapter_config.json").write_text("{}")


def test_planned_steps_matches_the_trainer():
    # 3000 examples / 16 per step = 188 per epoch, x2 epochs
    assert planned_steps(CFG, 3000) == 376
    assert planned_steps({**CFG, "max_steps": 40}, 3000) == 40


def test_no_snapshot_means_a_fresh_start(tmp_path):
    cfg, offset, path = _resume_plan(CFG, tmp_path, 3000)
    assert (offset, path) == (0, None)
    assert cfg["max_steps"] == -1


def test_latest_snapshot_wins_and_remaining_steps_are_computed(tmp_path):
    _snapshot(tmp_path, 100)
    _snapshot(tmp_path, 250, nested=True)  # PEFT's named-adapter layout
    assert _latest_snapshot(tmp_path)[0] == 250
    cfg, offset, path = _resume_plan(CFG, tmp_path, 3000)
    assert offset == 250
    assert path == tmp_path / "step-250"
    assert cfg["max_steps"] == 376 - 250
    assert cfg["warmup_ratio"] == 0.0


def test_a_finished_run_is_not_resumed(tmp_path):
    _snapshot(tmp_path, 376)
    assert _resume_plan(CFG, tmp_path, 3000)[1] == 0


def test_snapshot_dirs_without_an_adapter_are_ignored(tmp_path):
    (tmp_path / "step-50").mkdir()
    assert _latest_snapshot(tmp_path) is None
