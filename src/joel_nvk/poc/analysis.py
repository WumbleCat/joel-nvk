"""Turn per-checkpoint records into the table the plots and the write-up read.

Pure Python on purpose: this is the last step before a number reaches the
paper, and it must be checkable without a GPU.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

CSV_COLUMNS = [
    "arm",
    "seed",
    "step",
    "old_task",
    "new_task",
    "old_acc",
    "new_acc",
    "forgetting",
    "old_parse_fail",
    "new_parse_fail",
    "old_len",
    "new_len",
    "kl_old",
    "kl_old_se",
    "kl_old_rev",
    "kl_new",
    "kl_new_se",
    "kl_new_rev",
    "correct_to_wrong",
    "wrong_to_correct",
    "train_loss",
    "lr",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def checkpoint_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten checkpoint records into CSV rows.

    Forgetting is defined against the arm's own step-0 row, i.e. M1:
    ``forgetting = old_acc(M1) - old_acc(step)``. Positive means the old
    capability was lost.
    """
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_arm.setdefault(record["arm"], []).append(record)

    rows: list[dict[str, Any]] = []
    for arm, items in by_arm.items():
        items = sorted(items, key=lambda r: int(r["step"]))
        reference = next((r for r in items if int(r["step"]) == 0), None)
        ref_acc = reference["old"]["accuracy"] if reference else None
        for record in items:
            old, new = record["old"], record["new"]
            kl = record.get("kl", {})
            rows.append(
                {
                    "arm": arm,
                    "seed": record.get("seed"),
                    "step": int(record["step"]),
                    "old_task": old["task"],
                    "new_task": new["task"],
                    "old_acc": old["accuracy"],
                    "new_acc": new["accuracy"],
                    "forgetting": (ref_acc - old["accuracy"]) if ref_acc is not None else None,
                    "old_parse_fail": old["parse_failure_rate"],
                    "new_parse_fail": new["parse_failure_rate"],
                    "old_len": old["mean_completion_len"],
                    "new_len": new["mean_completion_len"],
                    "kl_old": _kl(kl, "kl_old", "kl"),
                    "kl_old_se": _kl(kl, "kl_old", "se"),
                    "kl_old_rev": _kl(kl, "kl_old_rev", "kl"),
                    "kl_new": _kl(kl, "kl_new", "kl"),
                    "kl_new_se": _kl(kl, "kl_new", "se"),
                    "kl_new_rev": _kl(kl, "kl_new_rev", "kl"),
                    "correct_to_wrong": record.get("transitions", {}).get("correct_to_wrong"),
                    "wrong_to_correct": record.get("transitions", {}).get("wrong_to_correct"),
                    "train_loss": record.get("train_loss"),
                    "lr": record.get("lr"),
                }
            )
    return rows


def _kl(kl: dict[str, Any], axis: str, field: str) -> float | None:
    entry = kl.get(axis)
    return None if not entry else entry.get(field)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in CSV_COLUMNS})


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Plain Pearson r; None when there is nothing to correlate."""
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    n = len(pairs)
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    syy = sum((y - my) ** 2 for _, y in pairs)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    return sxy / math.sqrt(sxx * syy)


def summarise_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The headline numbers for one arm: start, peak, end, and the KL axes."""
    rows = sorted(rows, key=lambda r: r["step"])
    if not rows:
        return {}
    first, last = rows[0], rows[-1]
    best_new = max(rows, key=lambda r: r["new_acc"])
    worst_old = min(rows, key=lambda r: r["old_acc"])
    return {
        "steps": [r["step"] for r in rows],
        "old_acc_start": first["old_acc"],
        "old_acc_end": last["old_acc"],
        "old_acc_min": worst_old["old_acc"],
        "old_acc_min_step": worst_old["step"],
        "new_acc_start": first["new_acc"],
        "new_acc_end": last["new_acc"],
        "new_acc_max": best_new["new_acc"],
        "new_acc_max_step": best_new["step"],
        "forgetting_end": last["forgetting"],
        "forgetting_max": max((r["forgetting"] or 0.0) for r in rows),
        "kl_old_end": last["kl_old"],
        "kl_new_end": last["kl_new"],
        "kl_old_vs_new_r": pearson(
            [r["kl_old"] for r in rows if r["step"] > 0],
            [r["kl_new"] for r in rows if r["step"] > 0],
        ),
        "forgetting_vs_kl_old_r": pearson(
            [r["kl_old"] for r in rows if r["step"] > 0],
            [r["forgetting"] for r in rows if r["step"] > 0],
        ),
        "forgetting_vs_kl_new_r": pearson(
            [r["kl_new"] for r in rows if r["step"] > 0],
            [r["forgetting"] for r in rows if r["step"] > 0],
        ),
        "correct_to_wrong_end": last["correct_to_wrong"],
        "wrong_to_correct_end": last["wrong_to_correct"],
    }


def flags(rows_by_arm: dict[str, list[dict[str, Any]]], m0_m1: dict[str, Any] | None) -> list[str]:
    """The 'unexpected result' checks from the brief, evaluated on the data."""
    out: list[str] = []
    if m0_m1:
        gain = m0_m1["m1"]["old"]["accuracy"] - m0_m1["m0"]["old"]["accuracy"]
        if gain <= 0:
            out.append(
                f"Task-A training did not improve Task A (gain {gain:+.3f}): stage B invalid"
            )
        pf = m0_m1["m1"]["old"]["parse_failure_rate"]
        if pf > 0.1:
            out.append(
                f"M1 parse failure rate on Task A is {pf:.1%}: "
                "format, not capability, may be moving"
            )
    for arm, rows in rows_by_arm.items():
        s = summarise_arm(rows)
        if not s:
            continue
        if s["new_acc_end"] <= s["new_acc_start"]:
            out.append(
                f"{arm}: Task B did not improve "
                f"({s['new_acc_start']:.3f} -> {s['new_acc_end']:.3f})"
            )
        if s["forgetting_max"] <= 0:
            out.append(f"{arm}: no forgetting observed at any checkpoint")
        r = s["kl_old_vs_new_r"]
        if r is not None and r > 0.98:
            out.append(
                f"{arm}: KL_old and KL_new nearly collinear (r={r:.3f}); axes may not separate"
            )
        late = [r for r in rows if r["step"] > 0]
        if late and all((r["new_parse_fail"] or 0) > 0.2 for r in late):
            out.append(f"{arm}: Task-B parse failures above 20% at every checkpoint: format drift")
    return out
