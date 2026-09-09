"""Turn a run's result records into a readable summary.

Reads only ``outputs/results/<run_id>.jsonl`` — the same rule the analysis code
follows, so a number printed here is a number that was actually recorded.
"""

import json
from pathlib import Path
from typing import Any

from joel_nvk.utils.paths import outputs_dir

STATE_LABELS = {
    "base": "base",
    "math": "LoRA(MATH)",
    "mmlu": "LoRA(MATH->MMLU)",
}
STATE_ORDER = ("base", "math", "mmlu")


def results_path(run_id: str) -> Path:
    return outputs_dir() / "results" / f"{run_id}.jsonl"


def load_records(run_id: str) -> list[dict[str, Any]]:
    path = results_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"No results for run {run_id!r} at {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect the last evaluation per state, plus the KL record."""
    evaluations: dict[str, dict[str, Any]] = {}
    training: dict[str, dict[str, Any]] = {}
    kl: dict[str, Any] = {}
    progress: list[dict[str, Any]] = []

    for record in records:
        if record.get("stage") == "progress":
            progress.append(record)
        elif "math" in record and "mmlu" in record:
            evaluations[record["state"]] = record
        elif "train" in record:
            training[record["state"]] = record["train"]
        elif record.get("stage") == "kl":
            kl = record

    return {"evaluations": evaluations, "training": training, "kl": kl, "progress": progress}


def _pct(value: float) -> str:
    return f"{100 * value:5.1f}%"


def format_summary(run_id: str, summary: dict[str, Any]) -> str:
    """Render the accuracy, forgetting and KL tables."""
    evaluations = summary["evaluations"]
    lines = [f"run: {run_id}", ""]

    lines.append("accuracy")
    lines.append(
        f"  {'state':<18} {'MATH 3-5':>9} {'MMLU':>9} {'MATH parse-fail':>16} {'MATH len':>9}"
    )
    for state in STATE_ORDER:
        record = evaluations.get(state)
        if not record:
            continue
        math, mmlu = record["math"], record["mmlu"]
        lines.append(
            f"  {STATE_LABELS[state]:<18} {_pct(math['accuracy']):>9} {_pct(mmlu['accuracy']):>9} "
            f"{_pct(math['parse_failure_rate']):>16} {math['mean_completion_len']:>9.0f}"
        )

    by_level = {
        state: record["math"].get("accuracy_by_level", {}) for state, record in evaluations.items()
    }
    levels = sorted({level for table in by_level.values() for level in table})
    if levels:
        lines += ["", "MATH accuracy by level"]
        lines.append("  " + f"{'state':<18}" + "".join(f"{'L' + level:>9}" for level in levels))
        for state in STATE_ORDER:
            if state not in by_level:
                continue
            row = "".join(
                f"{_pct(by_level[state].get(level, float('nan'))):>9}" for level in levels
            )
            lines.append(f"  {STATE_LABELS[state]:<18}{row}")

    if "math" in evaluations and "mmlu" in evaluations:
        before = evaluations["math"]["math"]["accuracy"]
        after = evaluations["mmlu"]["math"]["accuracy"]
        mmlu_before = evaluations["math"]["mmlu"]["accuracy"]
        mmlu_after = evaluations["mmlu"]["mmlu"]["accuracy"]
        math_change = 100 * (after - before)
        mmlu_change = 100 * (mmlu_after - mmlu_before)
        lines += [
            "",
            "effect of the second training stage (LoRA(MATH) -> after MMLU)",
            f"  MATH 3-5 : {_pct(before)} -> {_pct(after)}"
            f"   change {math_change:+.1f} points   (forgetting {-math_change:.1f} points)",
            f"  MMLU     : {_pct(mmlu_before)} -> {_pct(mmlu_after)}"
            f"   change {mmlu_change:+.1f} points",
        ]
        if "base" in evaluations:
            base_acc = evaluations["base"]["math"]["accuracy"]
            lines.append(
                f"  MATH 3-5 relative to the base model: {100 * (after - base_acc):+.1f} points"
            )

    kl = summary["kl"]
    if kl:
        lines += ["", "KL(policy || reference), exact token-level, teacher-forced"]
        lines.append(f"  {'pair':<20} {'kl_old (MATH)':>18} {'kl_new (MMLU)':>18}")
        for pair in kl.get("pairs", []):
            old = kl.get("kl_old", {}).get(pair, {})
            new = kl.get("kl_new", {}).get(pair, {})
            lines.append(f"  {pair:<20} {_fmt_kl(old):>18} {_fmt_kl(new):>18}")

    progress = summary.get("progress", [])
    if progress:
        lines += ["", "stages (every attempt, in order)"]
        for item in progress:
            marker = "ok    " if item.get("status") == "ok" else "FAILED"
            lines.append(
                f"  {item.get('timestamp', '')[:19]:<19} {marker} {item['stage_name']:<11}"
                f" {item.get('duration_s', 0):>7.0f}s"
                + (f"  {item['error']}" if item.get("error") else "")
            )

    training = summary["training"]
    if training:
        lines += ["", "training"]
        for state, info in training.items():
            lines.append(
                f"  {STATE_LABELS.get(state, state):<18} {info['steps']:>4} steps  "
                f"loss {info['train_loss']:.4f}  on {info['n_examples']} examples"
            )

    return "\n".join(lines)


def _fmt_kl(entry: dict[str, Any]) -> str:
    if not entry:
        return "-"
    return f"{entry['kl']:.4f} +/- {entry['se']:.4f}"
