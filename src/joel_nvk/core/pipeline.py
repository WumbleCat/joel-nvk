"""The pipeline-validation run.

Stages, in order:

1. ``prepare``          build and hash the frozen train/eval/probe sets
2. ``baseline``         evaluate the base model on MATH (levels 3-5) and MMLU
3. ``train_math``       LoRA SFT on MATH  -> adapter ``math``
4. ``eval_math``        evaluate the ``math`` adapter on both tasks
5. ``train_mmlu``       continue the same adapter on MMLU -> adapter ``mmlu``
6. ``eval_mmlu``        evaluate the ``mmlu`` adapter on both tasks (forgetting)
7. ``kl``               exact token-level KL between base / math / mmlu states

Stages are resumable: pass an existing ``run_id`` and a subset of stages, and the
adapters and frozen sets are picked up from disk.

This is a smoke test of the machinery, not the experiment. It runs one seed and
one drift point, so it can show that the plumbing produces sane numbers — it
cannot support a claim about methods.
"""

import hashlib
import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from joel_nvk.data.prompts import Example, render_prompt
from joel_nvk.utils.paths import data_dir, outputs_dir
from joel_nvk.utils.seeding import derive_seed, set_seed

logger = logging.getLogger(__name__)

STAGES = (
    "prepare",
    "baseline",
    "train_math",
    "eval_math",
    "train_mmlu",
    "eval_mmlu",
    "kl",
)

BASE_STATE = "base"
MATH_STATE = "math"
MMLU_STATE = "mmlu"


@dataclass
class Run:
    """Everything a stage needs to read its inputs and write its outputs."""

    run_id: str
    config: dict[str, Any]
    seed: int

    @property
    def checkpoints(self) -> Path:
        return outputs_dir() / "checkpoints" / self.run_id

    def adapter_path(self, state: str) -> Path:
        return self.checkpoints / state

    @property
    def results_file(self) -> Path:
        return outputs_dir() / "results" / f"{self.run_id}.jsonl"

    @property
    def predictions_file(self) -> Path:
        return outputs_dir() / "results" / f"{self.run_id}.predictions.jsonl"

    @property
    def manifest_file(self) -> Path:
        return outputs_dir() / "results" / f"{self.run_id}.manifest.json"

    def record(self, stage: str, payload: dict[str, Any]) -> None:
        """Append one immutable result record."""
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "run_id": self.run_id,
            "stage": stage,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **payload,
        }
        with self.results_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        logger.info("[%s] %s", stage, json.dumps(_loggable(payload)))

    def write_predictions(self, stage: str, state: str, predictions: list[dict[str, Any]]) -> None:
        self.predictions_file.parent.mkdir(parents=True, exist_ok=True)
        with self.predictions_file.open("a", encoding="utf-8") as handle:
            for item in predictions:
                handle.write(json.dumps({"stage": stage, "state": state, **item}) + "\n")


def _loggable(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky fields so the log line stays readable."""
    return {k: v for k, v in payload.items() if k != "predictions"}


def make_run_id(config: dict[str, Any], seed: int) -> str:
    model = str(config["model"]["id"]).split("/")[-1]
    return f"pipeline-{model}-s{seed}-{datetime.now():%Y%m%d-%H%M%S}"


def dataset_hash(examples: list[Example]) -> str:
    """Content hash of a frozen set — proves two stages used the same items."""
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.prompt.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(example.target.encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()[:16]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_manifest(run: Run, hashes: dict[str, str]) -> None:
    """Record commit, config, environment and data hashes before any training."""
    import torch

    dirty = bool(_git("status", "--porcelain"))
    if dirty:
        logger.warning("Working tree is dirty — this run is not citable")

    manifest = {
        "run_id": run.run_id,
        "seed": run.seed,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": dirty,
        "config": run.config,
        "dataset_hashes": hashes,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        import transformers

        manifest["transformers"] = transformers.__version__
    except ImportError:  # pragma: no cover - transformers is required to run
        pass

    run.manifest_file.parent.mkdir(parents=True, exist_ok=True)
    run.manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Manifest -> %s", run.manifest_file)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def build_datasets(config: dict[str, Any], seed: int) -> dict[str, list[Example]]:
    """Build the frozen train/eval sets. Deterministic in ``seed``."""
    from joel_nvk.data.math_dataset import assert_disjoint, load_math
    from joel_nvk.data.mmlu_dataset import load_mmlu

    math_cfg = config["data"]["math"]
    mmlu_cfg = config["data"]["mmlu"]

    math_train = load_math(
        math_cfg["train_split"],
        dataset=math_cfg["dataset"],
        subjects=math_cfg["subjects"],
        size=math_cfg["train_size"],
        seed=derive_seed(seed, "math_train"),
    )
    math_eval = load_math(
        math_cfg["eval_split"],
        dataset=math_cfg["dataset"],
        subjects=math_cfg["subjects"],
        levels=math_cfg["eval_levels"],
        size=math_cfg["eval_size"],
        seed=derive_seed(seed, "math_eval"),
    )
    assert_disjoint(math_train, math_eval)

    mmlu_train = load_mmlu(
        mmlu_cfg["train_split"],
        dataset=mmlu_cfg["dataset"],
        config=mmlu_cfg["config"],
        size=mmlu_cfg["train_size"],
        seed=derive_seed(seed, "mmlu_train"),
    )
    mmlu_eval = load_mmlu(
        mmlu_cfg["eval_split"],
        dataset=mmlu_cfg["dataset"],
        config=mmlu_cfg["config"],
        size=mmlu_cfg["eval_size"],
        seed=derive_seed(seed, "mmlu_eval"),
    )

    return {
        "math_train": math_train,
        "math_eval": math_eval,
        "mmlu_train": mmlu_train,
        "mmlu_eval": mmlu_eval,
    }


def freeze_datasets(datasets: dict[str, list[Example]]) -> dict[str, str]:
    """Write the frozen sets to ``data/processed/`` and return their hashes."""
    hashes = {}
    target = data_dir() / "processed"
    target.mkdir(parents=True, exist_ok=True)
    for name, examples in datasets.items():
        digest = dataset_hash(examples)
        hashes[name] = digest
        path = target / f"{name}-{digest}.jsonl"
        if not path.exists():
            with path.open("w", encoding="utf-8") as handle:
                for example in examples:
                    handle.write(
                        json.dumps(
                            {
                                "prompt": example.prompt,
                                "target": example.target,
                                "meta": example.meta,
                            }
                        )
                        + "\n"
                    )
        logger.info("%s: %d examples, hash %s", name, len(examples), digest)
    return hashes


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def _load_for_eval(config: dict[str, Any], adapters: dict[str, Path]) -> tuple[Any, Any]:
    from joel_nvk.models.loading import load_adapters, load_base_model, load_tokenizer

    model_cfg = config["model"]
    tokenizer = load_tokenizer(model_cfg["id"], padding_side="left")
    model = load_base_model(model_cfg["id"], device=model_cfg["device"], dtype=model_cfg["dtype"])
    model = load_adapters(model, adapters)
    return model, tokenizer


def _release(model: Any) -> None:
    import gc

    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_state(
    run: Run,
    stage: str,
    state: str,
    datasets: dict[str, list[Example]],
    adapters: dict[str, Path],
) -> dict[str, Any]:
    """Evaluate one model state on both tasks and record the result."""
    from joel_nvk.core.evaluate import evaluate_math, evaluate_mmlu
    from joel_nvk.models.loading import adapter_state

    eval_cfg = run.config["eval"]
    model, tokenizer = _load_for_eval(run.config, adapters)

    with adapter_state(model, state):
        math_result = evaluate_math(
            model,
            tokenizer,
            datasets["math_eval"],
            max_new_tokens=int(eval_cfg["math_max_new_tokens"]),
            batch_size=int(eval_cfg["batch_size"]),
        )
        mmlu_result = evaluate_mmlu(
            model,
            tokenizer,
            datasets["mmlu_eval"],
            max_new_tokens=int(eval_cfg["mmlu_max_new_tokens"]),
            batch_size=int(eval_cfg["batch_size"]),
        )

    run.write_predictions(stage, state, math_result.pop("predictions"))
    run.write_predictions(stage, state, mmlu_result.pop("predictions"))
    _release(model)

    payload = {"state": state, "math": math_result, "mmlu": mmlu_result}
    run.record(stage, payload)
    return payload


def stage_train(
    run: Run,
    stage: str,
    examples: list[Example],
    *,
    train_key: str,
    from_adapter: Path | None,
    to_state: str,
) -> dict[str, Any]:
    """Train the LoRA adapter, either fresh or continuing from a saved one."""
    from joel_nvk.models.loading import (
        attach_lora,
        load_adapter_for_training,
        load_base_model,
        load_tokenizer,
    )
    from joel_nvk.models.train import train_lora

    model_cfg = run.config["model"]
    tokenizer = load_tokenizer(model_cfg["id"], padding_side="right")
    model = load_base_model(model_cfg["id"], device=model_cfg["device"], dtype=model_cfg["dtype"])

    if from_adapter is None:
        model = attach_lora(model, run.config["lora"], adapter_name=to_state)
    else:
        model = load_adapter_for_training(model, from_adapter, adapter_name=to_state)

    summary = train_lora(
        model,
        tokenizer,
        examples,
        train_cfg=run.config["train"][train_key],
        output_dir=run.adapter_path(to_state),
        seed=run.seed,
        run_name=f"{run.run_id}-{to_state}",
    )
    _release(model)

    payload = {"state": to_state, "train": summary}
    run.record(stage, payload)
    return payload


def stage_kl(run: Run, datasets: dict[str, list[Example]]) -> dict[str, Any]:
    """Probe both KL axes between the three states.

    Continuations are generated once by the **base** model, so every state is
    scored on identical tokens.
    """
    from joel_nvk.core.generate import generate
    from joel_nvk.core.kl import build_probe, kl_between_states
    from joel_nvk.models.loading import adapter_state

    kl_cfg = run.config["kl"]
    adapters = {
        MATH_STATE: run.adapter_path(MATH_STATE),
        MMLU_STATE: run.adapter_path(MMLU_STATE),
    }
    missing = [name for name, path in adapters.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing adapters for the KL stage: {missing}")

    model, tokenizer = _load_for_eval(run.config, adapters)
    pairs = [
        (MATH_STATE, BASE_STATE),
        (MMLU_STATE, BASE_STATE),
        (MMLU_STATE, MATH_STATE),
    ]

    payload: dict[str, Any] = {"pairs": [f"{a}||{b}" for a, b in pairs]}
    # kl_old is measured where forgetting is measured (MATH); kl_new where the
    # second training happened (MMLU). They are never averaged together.
    for axis, key in (("kl_old", "math_eval"), ("kl_new", "mmlu_eval")):
        examples = datasets[key][: int(kl_cfg["probe_size"])]
        prompts = [render_prompt(tokenizer, ex.prompt) for ex in examples]

        with adapter_state(model, BASE_STATE):
            completions = generate(
                model,
                tokenizer,
                prompts,
                max_new_tokens=int(kl_cfg["max_new_tokens"]),
                batch_size=int(kl_cfg["generate_batch_size"]),
            )

        probe = build_probe(tokenizer, prompts, completions)
        payload[axis] = kl_between_states(
            model,
            tokenizer,
            probe,
            pairs,
            batch_size=int(kl_cfg["batch_size"]),
        )

    _release(model)
    run.record("kl", payload)
    return payload


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_pipeline(
    config: dict[str, Any],
    *,
    stages: list[str] | None = None,
    run_id: str | None = None,
) -> Run:
    """Execute the requested stages in order and return the run handle."""
    seed = int(config["project"]["seed"])
    set_seed(seed)

    selected = list(stages or STAGES)
    unknown = [stage for stage in selected if stage not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stages {unknown}; valid stages are {list(STAGES)}")
    selected.sort(key=STAGES.index)

    run = Run(run_id=run_id or make_run_id(config, seed), config=config, seed=seed)
    logger.info("Run %s — stages: %s", run.run_id, ", ".join(selected))

    datasets = build_datasets(config, seed)
    hashes = freeze_datasets(datasets)

    if "prepare" in selected:
        write_manifest(run, hashes)
        run.record(
            "prepare",
            {
                "dataset_hashes": hashes,
                "sizes": {name: len(items) for name, items in datasets.items()},
            },
        )

    if "baseline" in selected:
        evaluate_state(run, "baseline", BASE_STATE, datasets, adapters={})

    if "train_math" in selected:
        stage_train(
            run,
            "train_math",
            datasets["math_train"],
            train_key="math",
            from_adapter=None,
            to_state=MATH_STATE,
        )

    if "eval_math" in selected:
        evaluate_state(
            run,
            "eval_math",
            MATH_STATE,
            datasets,
            adapters={MATH_STATE: run.adapter_path(MATH_STATE)},
        )

    if "train_mmlu" in selected:
        stage_train(
            run,
            "train_mmlu",
            datasets["mmlu_train"],
            train_key="mmlu",
            from_adapter=run.adapter_path(MATH_STATE),
            to_state=MMLU_STATE,
        )

    if "eval_mmlu" in selected:
        evaluate_state(
            run,
            "eval_mmlu",
            MMLU_STATE,
            datasets,
            adapters={MMLU_STATE: run.adapter_path(MMLU_STATE)},
        )

    if "kl" in selected:
        stage_kl(run, datasets)

    logger.info("Results -> %s", run.results_file)
    return run
