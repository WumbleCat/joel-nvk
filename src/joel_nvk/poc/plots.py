"""The five PoC figures (plus one diagnostic), as PNG and SVG.

Static, publication-oriented matplotlib. Colour carries arm identity in a fixed
order (sft, self_sft, iter_sft) and never changes with which arms are present;
every line is also direct-labelled at its end so identity never rests on colour
alone. Every point is a checkpoint; step numbers annotate the KL and
learning-vs-forgetting views because there the x-axis is not time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ARM_ORDER = ("sft", "self_sft", "iter_sft")
ARM_COLOR = {"sft": "#2a78d6", "self_sft": "#eb6834", "iter_sft": "#1baf7a"}
ARM_LABEL = {"sft": "Vanilla SFT", "self_sft": "Self-SFT", "iter_sft": "Iterative-SFT"}
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 10,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "semibold",
            "axes.titlesize": 11,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )


def _by_arm(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["arm"], []).append(row)
    return {
        arm: sorted(grouped[arm], key=lambda r: r["step"]) for arm in ARM_ORDER if arm in grouped
    }


def _pct(ax: Any) -> None:
    from matplotlib.ticker import PercentFormatter

    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def _series(
    ax: Any,
    arms: dict[str, list[dict[str, Any]]],
    x: str,
    y: str,
    *,
    yerr: str | None = None,
    annotate_steps: bool = False,
    connect: bool = True,
) -> None:
    for arm, rows in arms.items():
        pts = [
            (r[x], r[y], r["step"], r.get(yerr) if yerr else None)
            for r in rows
            if r.get(x) is not None and r.get(y) is not None
        ]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        colour = ARM_COLOR[arm]
        if connect:
            ax.plot(
                xs,
                ys,
                color=colour,
                marker="o",
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                label=ARM_LABEL[arm],
                zorder=3,
            )
        else:
            ax.scatter(
                xs,
                ys,
                color=colour,
                s=42,
                edgecolor=SURFACE,
                linewidth=1.2,
                label=ARM_LABEL[arm],
                zorder=3,
            )
        if yerr:
            errs = [p[3] or 0.0 for p in pts]
            ax.errorbar(
                xs,
                ys,
                yerr=errs,
                fmt="none",
                ecolor=colour,
                elinewidth=1,
                capsize=2,
                alpha=0.6,
                zorder=2,
            )
        # Direct label at the line's end, so identity does not rest on colour.
        ax.annotate(
            ARM_LABEL[arm],
            (xs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK_2,
            fontsize=8.5,
            va="center",
        )
        if annotate_steps:
            for px, py, step, _ in pts:
                ax.annotate(
                    str(step),
                    (px, py),
                    xytext=(0, 7),
                    textcoords="offset points",
                    color=MUTED,
                    fontsize=7,
                    ha="center",
                )


def _finish(fig: Any, out_dir: Path, stem: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.tight_layout()
    fig.savefig(png, dpi=200)
    fig.savefig(svg)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return {"png": str(png), "svg": str(svg)}


def make_figures(
    rows: list[dict[str, Any]], out_dir: Path, *, task_a: str, task_b: str
) -> dict[str, dict[str, str]]:
    """Render every figure from checkpoint rows; returns name -> {png, svg}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    arms = _by_arm(rows)
    if not arms:
        logger.warning("No checkpoint rows — no figures")
        return {}
    multi = len(arms) > 1
    figures: dict[str, dict[str, str]] = {}

    # 1. New-task learning
    fig, ax = plt.subplots(figsize=(6.4, 4))
    _series(ax, arms, "step", "new_acc")
    ax.set(
        xlabel="training step",
        ylabel=f"{task_b} accuracy",
        title=f"Plot 1 — new-task learning ({task_b})",
    )
    _pct(ax)
    if multi:
        ax.legend(loc="lower right")
    figures["1_new_task_learning"] = _finish(fig, out_dir, "1_new_task_learning")

    # 2. Forgetting over training (accuracy view and forgetting view)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _series(axes[0], arms, "step", "old_acc")
    axes[0].set(
        xlabel="training step",
        ylabel=f"{task_a} accuracy",
        title=f"Plot 2a — old-task accuracy ({task_a})",
    )
    _pct(axes[0])
    _series(axes[1], arms, "step", "forgetting")
    axes[1].axhline(0, color=AXIS, linewidth=1)
    axes[1].set(
        xlabel="training step",
        ylabel="forgetting = acc_A(M1) − acc_A(step)",
        title="Plot 2b — forgetting over training",
    )
    _pct(axes[1])
    if multi:
        axes[0].legend(loc="lower left")
    figures["2_forgetting_over_training"] = _finish(fig, out_dir, "2_forgetting_over_training")

    # 3. Drift over training, one panel per axis
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, axis, prompts in ((axes[0], "kl_old", task_a), (axes[1], "kl_new", task_b)):
        _series(ax, arms, "step", axis, yerr=f"{axis}_se")
        ax.set(
            xlabel="training step",
            ylabel="KL(M1 ‖ checkpoint), nats/token",
            title=f"Plot 3 — drift on {prompts} prompts ({axis})",
        )
    if multi:
        axes[0].legend(loc="upper left")
    figures["3_drift_over_training"] = _finish(fig, out_dir, "3_drift_over_training")

    # 4. Forgetting vs KL, one panel per axis — the plot the later project turns on
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, axis, prompts in ((axes[0], "kl_old", task_a), (axes[1], "kl_new", task_b)):
        _series(ax, arms, axis, "forgetting", annotate_steps=True)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set(
            xlabel=f"{axis}: KL(M1 ‖ checkpoint) on {prompts} prompts",
            ylabel="old-task forgetting",
            title=f"Plot 4 — forgetting vs {axis}",
        )
        _pct(ax)
    if multi:
        axes[0].legend(loc="upper left")
    figures["4_forgetting_vs_kl"] = _finish(fig, out_dir, "4_forgetting_vs_kl")

    # 5. Learning vs forgetting (the traditional matched-new-task view)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    _series(ax, arms, "new_acc", "forgetting", annotate_steps=True)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set(
        xlabel=f"{task_b} accuracy",
        ylabel="old-task forgetting",
        title="Plot 5 — learning vs forgetting",
    )
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _pct(ax)
    if multi:
        ax.legend(loc="upper left")
    figures["5_learning_vs_forgetting"] = _finish(fig, out_dir, "5_learning_vs_forgetting")

    # 6. Diagnostic: are the two axes just the same number?
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    _series(ax, arms, "kl_new", "kl_old", annotate_steps=True)
    lim = max([r.get("kl_new") or 0 for r in rows] + [r.get("kl_old") or 0 for r in rows] + [1e-6])
    ax.plot([0, lim], [0, lim], color=AXIS, linewidth=1, linestyle="--", zorder=1)
    ax.set(
        xlabel=f"kl_new ({task_b} prompts)",
        ylabel=f"kl_old ({task_a} prompts)",
        title="Diagnostic — kl_old vs kl_new",
    )
    if multi:
        ax.legend(loc="upper left")
    figures["6_kl_old_vs_kl_new"] = _finish(fig, out_dir, "6_kl_old_vs_kl_new")

    logger.info("Figures -> %s", out_dir)
    return figures
