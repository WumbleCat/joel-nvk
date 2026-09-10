"""The proof-of-concept pipeline.

    M0  base model
     |  stage B: LoRA on Task A, merged into the weights
    M1  old-capability reference checkpoint
     |  stage C: new-task post-training on Task B, one arm at a time
    M2  checkpoints along the way

Stages (``STAGES`` below), each resumable and each runnable in its own process:

``probe``         M0 on every candidate task — headroom, verifier, failure examples
``prepare``       freeze and hash the train/eval sets, write the manifest
``train_old``     M0 -> M1: LoRA on Task A, then merge and save M1 as a full model
``eval_m0_m1``    M0 and M1 on Task A, Task B and the controls; validity check
``build_self``    M1's own verified Task-B answers (the Self-SFT dataset)
``train_new``     M1 -> M2 for one arm, snapshotting adapters at ``save_steps``
``eval_ckpts``    every checkpoint of one arm: accuracy on A and B, KL vs M1
``aggregate``     checkpoints.csv, transitions, figures, RESULTS.md

Forgetting and KL are both measured against **M1**, the model whose acquired
capability we are trying to preserve. Stage-C adapters attach to the merged M1,
so disabling them recovers M1 exactly and the KL reference costs no extra memory.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from joel_nvk.core.pipeline import _git, dataset_hash
from joel_nvk.core.task_eval import evaluate_task, transitions
from joel_nvk.data.prompts import Example, render_prompt
from joel_nvk.data.tasks import Task, get_task
from joel_nvk.utils.paths import outputs_dir
from joel_nvk.utils.seeding import derive_seed, set_seed

logger = logging.getLogger(__name__)

STAGES = (
    "probe",
    "prepare",
    "train_old",
    "eval_m0_m1",
    "build_self",
    "train_new",
    "eval_ckpts",
    "aggregate",
)
ARMS = ("sft", "self_sft", "iter_sft")
PER_ARM_STAGES = {"train_new", "eval_ckpts"}

REF_STATE = "base"  # adapter disabled == M1 (the loader's name for that state)
CKPT_STATE = "ckpt"


@dataclass
class PocRun:
    run_id: str
    config: dict[str, Any]
    seed: int

    @property
    def root(self) -> Path:
        return outputs_dir() / "poc" / self.run_id

    @property
    def m1_dir(self) -> Path:
        return self.root / "models" / "M1"

    @property
    def old_adapter_dir(self) -> Path:
        return self.root / "models" / "old_task_adapter"

    def arm_dir(self, arm: str) -> Path:
        return self.root / "arms" / arm

    @property
    def records_file(self) -> Path:
        return self.root / "checkpoints.jsonl"

    @property
    def log_file(self) -> Path:
        return outputs_dir() / "logs" / f"{self.run_id}.log"

    def path(self, *parts: str) -> Path:
        target = self.root.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_json(self, name: str, payload: Any) -> Path:
        target = self.path(name)
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return target

    def read_json(self, name: str) -> Any:
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def write_predictions(self, name: str, predictions: list[dict[str, Any]]) -> Path:
        target = self.path("predictions", f"{name}.jsonl")
        with target.open("w", encoding="utf-8") as handle:
            for item in predictions:
                handle.write(json.dumps(item) + "\n")
        return target

    def read_predictions(self, name: str) -> list[dict[str, Any]]:
        target = self.root / "predictions" / f"{name}.jsonl"
        with target.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def append_record(self, payload: dict[str, Any]) -> None:
        self.records_file.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **payload,
        }
        with self.records_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")


def make_run_id(config: dict[str, Any]) -> str:
    model = str(config["model"]["id"]).split("/")[-1]
    return f"poc-{model}-{datetime.now():%Y%m%d-%H%M%S}"


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #


def _cfg(run: PocRun) -> dict[str, Any]:
    return run.config["poc"]


def _base_model_id(run: PocRun) -> str:
    return str(run.config["model"]["id"])


def _load(
    model_path: str, run: PocRun, *, padding_side: str, adapters: dict[str, Path] | None = None
) -> tuple[Any, Any]:
    from joel_nvk.models.loading import load_adapters, load_base_model, load_tokenizer

    model_cfg = run.config["model"]
    tokenizer = load_tokenizer(model_path, padding_side=padding_side)
    model = load_base_model(
        model_path, device=model_cfg.get("device", "auto"), dtype=model_cfg.get("dtype", "auto")
    )
    if adapters:
        model = load_adapters(model, adapters)
    return model, tokenizer


def _release(model: Any) -> None:
    import gc

    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def _load_split(task: Task, split: str, size: int | None, seed: int, label: str) -> list[Example]:
    return task.load(split, size=size, seed=derive_seed(seed, label))


def build_sets(run: PocRun) -> dict[str, list[Example]]:
    """The frozen sets for the whole run. Deterministic in the seed."""
    cfg = _cfg(run)
    sizes = cfg["sizes"]
    task_a, task_b = get_task(cfg["task_a"]), get_task(cfg["task_b"])
    if task_a.train_split is None or task_b.train_split is None:
        raise ValueError("Both Task A and Task B need a training split")

    a_train = _load_split(task_a, task_a.train_split, sizes["a_train"], run.seed, "a_train")
    for extra_name in cfg.get("task_a_train_extra", []) or []:
        extra = get_task(extra_name)
        remaining = sizes["a_train"] - len(a_train)
        if remaining <= 0:
            break
        a_train += _load_split(
            extra, extra.train_split or "train", remaining, run.seed, f"a_train_{extra_name}"
        )
    a_eval = _load_split(task_a, task_a.eval_split, sizes["a_eval"], run.seed, "a_eval")
    b_train = _load_split(task_b, task_b.train_split, sizes["b_train"], run.seed, "b_train")
    b_eval = _load_split(task_b, task_b.eval_split, sizes["b_eval"], run.seed, "b_eval")

    # Held-out means held out: no evaluation prompt may appear in training.
    _assert_disjoint(a_train, a_eval, "Task A")
    _assert_disjoint(b_train, b_eval, "Task B")

    sets = {"a_train": a_train, "a_eval": a_eval, "b_train": b_train, "b_eval": b_eval}
    for name in cfg.get("controls", []) or []:
        control = get_task(name)
        sets[f"control_{name}"] = _load_split(
            control, control.eval_split, sizes["control_eval"], run.seed, f"control_{name}"
        )
    return sets


def _assert_disjoint(train: list[Example], evaluation: list[Example], label: str) -> None:
    seen = {ex.prompt for ex in train}
    overlap = sum(ex.prompt in seen for ex in evaluation)
    if overlap:
        raise ValueError(f"{label}: {overlap} evaluation prompts also appear in training")


def _read_examples(path: Path) -> list[Example]:
    with path.open("r", encoding="utf-8") as handle:
        return [Example(**json.loads(line)) for line in handle if line.strip()]


def _write_examples(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(
                json.dumps({"prompt": ex.prompt, "target": ex.target, "meta": ex.meta}) + "\n"
            )


def frozen_sets(run: PocRun) -> dict[str, list[Example]]:
    """Load the sets frozen by ``prepare``; build them if this is the first stage."""
    manifest = run.root / "data" / "sets.json"
    if manifest.exists():
        names = json.loads(manifest.read_text(encoding="utf-8"))["sets"]
        return {name: _read_examples(run.root / "data" / f"{name}.jsonl") for name in names}
    sets = build_sets(run)
    for name, examples in sets.items():
        _write_examples(run.root / "data" / f"{name}.jsonl", examples)
    run.write_json(
        "data/sets.json",
        {
            "sets": list(sets),
            "sizes": {k: len(v) for k, v in sets.items()},
            "hashes": {k: dataset_hash(v) for k, v in sets.items()},
        },
    )
    return sets


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def stage_probe(run: PocRun) -> dict[str, Any]:
    """M0 on each candidate task: accuracy, parse failures, headroom, failure examples."""
    cfg = _cfg(run)["probe"]
    model, tokenizer = _load(_base_model_id(run), run, padding_side="left")
    report: dict[str, Any] = {}
    for name in cfg["tasks"]:
        task = get_task(name)
        examples = task.load(
            task.eval_split, size=int(cfg["size"]), seed=derive_seed(run.seed, f"probe_{name}")
        )
        result = evaluate_task(
            model, tokenizer, task, examples, batch_size=int(_cfg(run)["eval"]["batch_size"])
        )
        failures = [p for p in result["predictions"] if not p["correct"]][
            : int(cfg.get("failure_examples", 10))
        ]
        run.write_predictions(f"probe_{name}", result.pop("predictions"))
        chance = 1.0 / examples[0].meta.get("n_choices", 1) if task.kind == "mc" else 0.0
        report[name] = {
            **result,
            "capability": task.capability,
            "kind": task.kind,
            "trainable": task.trainable and task.train_split is not None,
            "train_split": task.train_split,
            "chance": chance,
            "headroom": 1.0 - result["accuracy"],
            "note": task.note,
            "failures": [
                {
                    "gold": f["gold"],
                    "predicted": f["predicted"],
                    "completion": f["completion"][:400],
                }
                for f in failures
            ],
        }
    _release(model)
    run.write_json("probe/probe.json", report)
    (run.root / "probe" / "probe.md").write_text(probe_markdown(report), encoding="utf-8")
    return report


def probe_markdown(report: dict[str, Any]) -> str:
    header = "| task | capability | kind | acc (M0) | chance | headroom | parse-fail | mean len"
    lines = [
        header + " | trainable | n |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in report.items():
        cells = [
            name,
            r["capability"],
            r["kind"],
            f"{r['accuracy']:.3f}",
            f"{r['chance']:.2f}",
            f"{r['headroom']:.3f}",
            f"{r['parse_failure_rate']:.3f}",
            f"{r['mean_completion_len']:.0f}",
            "yes" if r["trainable"] else "no",
            str(r["n"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def stage_prepare(run: PocRun) -> dict[str, Any]:
    sets = frozen_sets(run)
    import torch

    manifest = {
        "run_id": run.run_id,
        "seed": run.seed,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config": run.config,
        "sizes": {k: len(v) for k, v in sets.items()},
        "hashes": {k: dataset_hash(v) for k, v in sets.items()},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    run.write_json("configs/manifest.json", manifest)
    run.write_json("configs/config.json", run.config)
    return manifest


def stage_train_old(run: PocRun) -> dict[str, Any]:
    """M0 -> M1: LoRA on Task A, merged into the weights and saved as a full model."""
    from joel_nvk.models.loading import attach_lora, merge_and_save
    from joel_nvk.models.train import train_lora

    sets = frozen_sets(run)
    model, tokenizer = _load(_base_model_id(run), run, padding_side="right")
    model = attach_lora(model, run.config["lora"], adapter_name="old_task")
    summary = train_lora(
        model,
        tokenizer,
        sets["a_train"],
        train_cfg=run.config["train"]["old"],
        output_dir=run.old_adapter_dir,
        seed=run.seed,
        run_name=f"{run.run_id}-old",
    )
    merge_and_save(model, tokenizer, run.m1_dir)
    _release(model)
    run.write_json("models/train_old.json", summary)
    return summary


def _eval_sets(
    run: PocRun,
    model: Any,
    tokenizer: Any,
    sets: dict[str, list[Example]],
    label: str,
    *,
    include_controls: bool,
) -> dict[str, Any]:
    cfg = _cfg(run)
    batch = int(cfg["eval"]["batch_size"])
    task_a, task_b = get_task(cfg["task_a"]), get_task(cfg["task_b"])
    out: dict[str, Any] = {}
    for key, task in (("old", task_a), ("new", task_b)):
        result = evaluate_task(
            model, tokenizer, task, sets["a_eval" if key == "old" else "b_eval"], batch_size=batch
        )
        run.write_predictions(f"{label}_{key}", result.pop("predictions"))
        out[key] = result
    if include_controls:
        for name in cfg.get("controls", []) or []:
            result = evaluate_task(
                model, tokenizer, get_task(name), sets[f"control_{name}"], batch_size=batch
            )
            run.write_predictions(f"{label}_control_{name}", result.pop("predictions"))
            out[f"control_{name}"] = result
    return out


def stage_eval_m0_m1(run: PocRun) -> dict[str, Any]:
    """Both reference checkpoints on everything, then the stage-B validity check."""
    sets = frozen_sets(run)
    results: dict[str, Any] = {}
    for label, path in (("m0", _base_model_id(run)), ("m1", str(run.m1_dir))):
        model, tokenizer = _load(path, run, padding_side="left")
        results[label] = _eval_sets(run, model, tokenizer, sets, label, include_controls=True)
        _release(model)

    gain = results["m1"]["old"]["accuracy"] - results["m0"]["old"]["accuracy"]
    threshold = float(_cfg(run)["validity"]["min_old_task_gain"])
    results["validity"] = {
        "old_task_gain": gain,
        "min_old_task_gain": threshold,
        "stage_b_valid": gain >= threshold,
        "old_task_transitions_m0_to_m1": transitions(
            run.read_predictions("m0_old"), run.read_predictions("m1_old")
        ),
    }
    run.write_json("eval/m0_m1.json", results)
    if not results["validity"]["stage_b_valid"]:
        logger.error(
            "STAGE B INVALID: Task A gain %+.3f < %.3f — stage C will not run", gain, threshold
        )
    return results


def stage_build_self(run: PocRun) -> dict[str, Any]:
    """M1's own verified Task-B answers."""
    from joel_nvk.models.self_sft import build_self_sft_dataset

    cfg = _cfg(run)["self_sft"]
    sets = frozen_sets(run)
    model, tokenizer = _load(str(run.m1_dir), run, padding_side="left")
    kept, stats = build_self_sft_dataset(
        model,
        tokenizer,
        get_task(_cfg(run)["task_b"]),
        sets["b_train"],
        samples_per_prompt=int(cfg["samples_per_prompt"]),
        temperature=float(cfg["temperature"]),
        batch_size=int(cfg.get("generate_batch_size", 8)),
        seed=derive_seed(run.seed, "self_sft"),
        keep_per_prompt=int(cfg.get("keep_per_prompt", 1)),
    )
    _release(model)
    _write_examples(run.root / "data" / "self_sft.jsonl", kept)
    run.write_json("data/self_sft_stats.json", stats)
    if stats["n_kept"] < 0.25 * stats["n_prompts"]:
        logger.warning(
            "Self-SFT retained only %d/%d prompts — viability is doubtful",
            stats["n_kept"],
            stats["n_prompts"],
        )
    return stats


def _require_valid(run: PocRun) -> None:
    path = run.root / "eval" / "m0_m1.json"
    if not path.exists():
        raise FileNotFoundError("Run eval_m0_m1 before stage C")
    if not run.read_json("eval/m0_m1.json")["validity"]["stage_b_valid"]:
        raise RuntimeError("Stage B is invalid (Task A did not improve); refusing to start stage C")


def stage_train_new(run: PocRun, arm: str) -> dict[str, Any]:
    """M1 -> M2 for one arm, snapshotting the adapter at ``save_steps``."""
    from joel_nvk.models.loading import attach_lora
    from joel_nvk.models.train import train_lora

    _require_valid(run)
    cfg = _cfg(run)
    sets = frozen_sets(run)
    save_steps = [int(s) for s in cfg["save_steps"]]
    arm_dir = run.arm_dir(arm)
    train_cfg = dict(run.config["train"]["new"])

    if arm == "sft":
        data = sets["b_train"]
    elif arm == "self_sft":
        data = _read_examples(run.root / "data" / "self_sft.jsonl")
    elif arm == "iter_sft":
        return _train_iter_sft(run, arm_dir, sets, train_cfg, save_steps)
    else:
        raise ValueError(f"Unknown arm {arm!r}; known: {ARMS}")

    model, tokenizer = _load(str(run.m1_dir), run, padding_side="right")
    model = attach_lora(model, run.config["lora"], adapter_name="new_task")
    summary = train_lora(
        model,
        tokenizer,
        data,
        train_cfg=train_cfg,
        output_dir=arm_dir,
        seed=run.seed,
        run_name=f"{run.run_id}-{arm}",
        save_steps=save_steps,
    )
    _release(model)
    summary["arm"] = arm
    summary["data_size"] = len(data)
    run.write_json(f"arms/{arm}/train.json", summary)
    return summary


def _train_iter_sft(
    run: PocRun,
    arm_dir: Path,
    sets: dict[str, list[Example]],
    train_cfg: dict[str, Any],
    save_steps: list[int],
) -> dict[str, Any]:
    """Regenerate verified data from the *current* policy at the start of each round."""
    from joel_nvk.models.loading import attach_lora, load_adapter_for_training, resolve_adapter_dir
    from joel_nvk.models.self_sft import build_self_sft_dataset
    from joel_nvk.models.train import train_lora

    cfg = _cfg(run)
    rounds = int(cfg["iter_sft"]["rounds"])
    self_cfg = cfg["self_sft"]
    task_b = get_task(cfg["task_b"])
    per_round = dict(train_cfg)
    per_round["epochs"] = max(float(train_cfg.get("epochs", 1)) / rounds, 1e-6)
    if int(train_cfg.get("max_steps", -1)) > 0:
        per_round["max_steps"] = max(int(train_cfg["max_steps"]) // rounds, 1)

    offset = 0
    summaries = []
    previous: Path | None = None
    for round_index in range(rounds):
        # Generation uses the current policy (M1 + adapter so far).
        adapters = {"new_task": previous} if previous else None
        model, tokenizer = _load(str(run.m1_dir), run, padding_side="left", adapters=adapters)
        kept, stats = build_self_sft_dataset(
            model,
            tokenizer,
            task_b,
            sets["b_train"],
            samples_per_prompt=int(self_cfg["samples_per_prompt"]),
            temperature=float(self_cfg["temperature"]),
            batch_size=int(self_cfg.get("generate_batch_size", 8)),
            seed=derive_seed(run.seed, "iter_sft", round_index),
            keep_per_prompt=int(self_cfg.get("keep_per_prompt", 1)),
        )
        _release(model)
        _write_examples(arm_dir / f"round-{round_index}-data.jsonl", kept)

        model, tokenizer = _load(str(run.m1_dir), run, padding_side="right")
        if previous is None:
            model = attach_lora(model, run.config["lora"], adapter_name="new_task")
        else:
            model = load_adapter_for_training(model, previous, adapter_name="new_task")
        local_steps = [s - offset for s in save_steps if s > offset]
        round_dir = arm_dir / f"round-{round_index}"
        summary = train_lora(
            model,
            tokenizer,
            kept,
            train_cfg=per_round,
            output_dir=round_dir,
            seed=derive_seed(run.seed, "iter_sft_train", round_index),
            run_name=f"{run.run_id}-iter{round_index}",
            save_steps=local_steps,
        )
        _release(model)
        # Re-key this round's snapshots to global steps so the ladder reads like the other arms.
        for local in summary["saved_steps"]:
            src = resolve_adapter_dir(round_dir / f"step-{local}")
            dst = arm_dir / f"step-{local + offset}"
            if not dst.exists():
                shutil.copytree(src, dst)
        offset += int(summary["steps"])
        summary["round"] = round_index
        summary["self_sft_stats"] = stats
        summaries.append(summary)
        previous = round_dir

    final = arm_dir / f"step-{offset}"
    if previous is not None and not final.exists():
        shutil.copytree(resolve_adapter_dir(previous), final)
    result = {
        "arm": "iter_sft",
        "rounds": summaries,
        "steps": offset,
        "saved_steps": sorted({int(p.name.split("-")[1]) for p in arm_dir.glob("step-*")}),
    }
    run.write_json("arms/iter_sft/train.json", result)
    return result


def _discover_steps(arm_dir: Path) -> list[int]:
    steps = sorted(int(p.name.split("-", 1)[1]) for p in arm_dir.glob("step-*") if p.is_dir())
    if not steps:
        raise FileNotFoundError(f"No step-* checkpoints under {arm_dir}")
    return steps


def _kl_probe(
    run: PocRun, model: Any, tokenizer: Any, sets: dict[str, list[Example]]
) -> dict[str, list[dict[str, Any]]]:
    """Continuations generated once by M1 on both prompt distributions, then frozen."""
    from joel_nvk.core.generate import generate
    from joel_nvk.core.kl import build_probe
    from joel_nvk.models.loading import adapter_state

    cfg = _cfg(run)["kl"]
    cache = run.root / "kl" / "probe.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    probes = {}
    for axis, key in (("old", "a_eval"), ("new", "b_eval")):
        examples = sets[key][: int(cfg["probe_size"])]
        prompts = [render_prompt(tokenizer, ex.prompt) for ex in examples]
        with adapter_state(model, REF_STATE):
            completions = generate(
                model,
                tokenizer,
                prompts,
                max_new_tokens=int(cfg["max_new_tokens"]),
                batch_size=int(cfg.get("generate_batch_size", 8)),
            )
        probes[axis] = build_probe(tokenizer, prompts, completions)
    run.write_json("kl/probe.json", probes)
    return probes


def stage_eval_ckpts(run: PocRun, arm: str) -> list[dict[str, Any]]:
    """Every checkpoint of one arm: accuracy on A and B, transitions vs M1, KL vs M1."""
    from joel_nvk.core.kl import kl_between_states

    cfg = _cfg(run)
    sets = frozen_sets(run)
    arm_dir = run.arm_dir(arm)
    steps = [0] + _discover_steps(arm_dir)
    train_info = (
        run.read_json(f"arms/{arm}/train.json") if (arm_dir / "train.json").exists() else {}
    )
    loss_by_step = {int(p["step"]): p for p in train_info.get("loss_curve", [])}
    done = {int(r["step"]) for r in _records_for(run, arm)}
    m1_old = run.read_predictions("m1_old")

    rows = []
    for step in steps:
        if step in done:
            logger.info("%s step %d already evaluated — skipping", arm, step)
            continue
        adapters = {CKPT_STATE: arm_dir / f"step-{step}"} if step > 0 else None
        model, tokenizer = _load(str(run.m1_dir), run, padding_side="left", adapters=adapters)
        label = f"{arm}_step{step}"
        # Accuracy: with the adapter active (step 0 has none, i.e. M1 itself).
        evals = _eval_sets(run, model, tokenizer, sets, label, include_controls=(step == steps[-1]))
        trans = transitions(m1_old, run.read_predictions(f"{label}_old"))

        kl: dict[str, Any] = {}
        if step > 0:
            probes = _kl_probe(run, model, tokenizer, sets)
            for axis in ("old", "new"):
                both = kl_between_states(
                    model,
                    tokenizer,
                    probes[axis],
                    [(REF_STATE, CKPT_STATE), (CKPT_STATE, REF_STATE)],
                    batch_size=int(cfg["kl"]["batch_size"]),
                )
                # Primary direction KL(M1 || ckpt): expectation under the model
                # whose capability we are preserving. The reverse is kept too.
                kl[f"kl_{axis}"] = both[f"{REF_STATE}||{CKPT_STATE}"]
                kl[f"kl_{axis}_rev"] = both[f"{CKPT_STATE}||{REF_STATE}"]
        else:
            zero = {"kl": 0.0, "se": 0.0, "n_prompts": 0, "n_tokens": 0}
            kl = {"kl_old": zero, "kl_old_rev": zero, "kl_new": zero, "kl_new_rev": zero}
        _release(model)

        nearest = max((s for s in loss_by_step if s <= step), default=None)
        record = {
            "arm": arm,
            "seed": run.seed,
            "step": step,
            "old": evals["old"],
            "new": evals["new"],
            "controls": {k: v for k, v in evals.items() if k.startswith("control_")},
            "transitions": trans,
            "kl": kl,
            "train_loss": loss_by_step[nearest]["loss"] if nearest is not None else None,
            "lr": loss_by_step[nearest]["lr"] if nearest is not None else None,
        }
        run.append_record(record)
        rows.append(record)
        logger.info(
            "%s step %d: old %.3f new %.3f kl_old %.4f kl_new %.4f",
            arm,
            step,
            evals["old"]["accuracy"],
            evals["new"]["accuracy"],
            kl["kl_old"]["kl"],
            kl["kl_new"]["kl"],
        )
    return rows


def _records_for(run: PocRun, arm: str | None = None) -> list[dict[str, Any]]:
    from joel_nvk.poc.analysis import load_jsonl

    records = load_jsonl(run.records_file)
    return [r for r in records if arm is None or r["arm"] == arm]


def stage_aggregate(run: PocRun) -> dict[str, Any]:
    """checkpoints.csv, arm summaries, flags, figures and RESULTS.md."""
    from joel_nvk.poc.analysis import checkpoint_rows, flags, summarise_arm, write_csv
    from joel_nvk.poc.plots import make_figures
    from joel_nvk.poc.report import write_results_md

    rows = checkpoint_rows(_records_for(run))
    write_csv(rows, run.root / "checkpoints.csv")
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    m0_m1 = (
        run.read_json("eval/m0_m1.json") if (run.root / "eval" / "m0_m1.json").exists() else None
    )
    summary = {
        "arms": {arm: summarise_arm(items) for arm, items in by_arm.items()},
        "flags": flags(by_arm, m0_m1),
        "m0_m1": m0_m1,
        "self_sft_stats": run.read_json("data/self_sft_stats.json")
        if (run.root / "data" / "self_sft_stats.json").exists()
        else None,
        "probe": run.read_json("probe/probe.json")
        if (run.root / "probe" / "probe.json").exists()
        else None,
        "train_old": run.read_json("models/train_old.json")
        if (run.root / "models" / "train_old.json").exists()
        else None,
    }
    run.write_json("summary.json", summary)
    figures = make_figures(
        rows, run.root / "figures", task_a=_cfg(run)["task_a"], task_b=_cfg(run)["task_b"]
    )
    write_results_md(run, rows, summary, figures)
    return summary


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_stage(run: PocRun, stage: str, arm: str | None = None) -> Any:
    started = time.monotonic()
    label = f"{stage}[{arm}]" if arm else stage
    result: Any
    logger.info("==== %s — started", label)
    try:
        if stage == "probe":
            result = stage_probe(run)
        elif stage == "prepare":
            result = stage_prepare(run)
        elif stage == "train_old":
            result = stage_train_old(run)
        elif stage == "eval_m0_m1":
            result = stage_eval_m0_m1(run)
        elif stage == "build_self":
            result = stage_build_self(run)
        elif stage == "train_new":
            result = stage_train_new(run, arm or "sft")
        elif stage == "eval_ckpts":
            result = stage_eval_ckpts(run, arm or "sft")
        elif stage == "aggregate":
            result = stage_aggregate(run)
        else:
            raise ValueError(f"Unknown stage {stage!r}; valid: {STAGES}")
    except Exception as exc:
        duration = time.monotonic() - started
        run.append_record(
            {
                "progress": True,
                "stage": label,
                "status": "failed",
                "duration_s": round(duration, 1),
                "error": repr(exc),
            }
        )
        logger.error("==== %s FAILED after %.0fs: %r", label, duration, exc)
        raise
    duration = time.monotonic() - started
    run.append_record(
        {"progress": True, "stage": label, "status": "ok", "duration_s": round(duration, 1)}
    )
    logger.info("==== %s — done in %.0fs", label, duration)
    return result


def plan(
    config: dict[str, Any], stages: list[str] | None, arms: list[str] | None
) -> list[tuple[str, str | None]]:
    """Expand a stage selection into (stage, arm) steps in pipeline order."""
    selected = list(stages or STAGES)
    unknown = [s for s in selected if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stages {unknown}; valid: {list(STAGES)}")
    selected.sort(key=STAGES.index)
    chosen_arms = list(arms or config["poc"].get("arms", ["sft"]))
    bad = [a for a in chosen_arms if a not in ARMS]
    if bad:
        raise ValueError(f"Unknown arms {bad}; valid: {list(ARMS)}")

    steps: list[tuple[str, str | None]] = []
    for stage in selected:
        if stage in PER_ARM_STAGES:
            steps += [(stage, arm) for arm in chosen_arms]
        elif stage == "build_self" and not ({"self_sft"} & set(chosen_arms)):
            continue  # nothing needs the self-SFT dataset
        else:
            steps.append((stage, None))
    return steps


def run_poc(
    config: dict[str, Any],
    *,
    stages: list[str] | None = None,
    arms: list[str] | None = None,
    run_id: str | None = None,
    low_memory: bool = False,
    env_name: str = "poc",
) -> PocRun:
    """Run the plan in-process, or one subprocess per step when ``low_memory``."""
    seed = int(config["project"]["seed"])
    set_seed(seed)
    run = PocRun(run_id=run_id or make_run_id(config), config=config, seed=seed)
    steps = plan(config, stages, arms)
    logger.info("Run %s — %s", run.run_id, ", ".join(f"{s}[{a}]" if a else s for s, a in steps))

    for stage, arm in steps:
        if low_memory:
            _run_in_subprocess(run, stage, arm, env_name)
        else:
            run_stage(run, stage, arm)
    return run


def _run_in_subprocess(run: PocRun, stage: str, arm: str | None, env_name: str) -> None:
    """One process per step so each stage starts from a clean memory footprint."""
    command = [
        sys.executable,
        "-m",
        "joel_nvk.poc.cli",
        "--env",
        env_name,
        "--run-id",
        run.run_id,
        "--stages",
        stage,
    ]
    if arm:
        command += ["--arms", arm]
    logger.info("$ %s", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{stage}[{arm}] failed in subprocess (exit {completed.returncode})")
