"""RESULTS.md for a run, written from the recorded numbers and nothing else."""

# ruff: noqa: E501  — Markdown table rows and sentences are clearer on one line.

from __future__ import annotations

from pathlib import Path
from typing import Any

ARM_LABEL = {"sft": "Vanilla SFT", "self_sft": "Self-SFT", "iter_sft": "Iterative-SFT"}


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _num(x: float | None, digits: int = 4) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _answer(question: str, verdict: str, evidence: str) -> str:
    return f"**{question}** {verdict}\n\n{evidence}\n"


def write_results_md(
    run: Any,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    figures: dict[str, dict[str, str]],
) -> Path:
    cfg = run.config["poc"]
    task_a, task_b = cfg["task_a"], cfg["task_b"]
    m0m1 = summary.get("m0_m1") or {}
    arms = summary.get("arms", {})
    lines: list[str] = [f"# Results — `{run.run_id}`", ""]
    lines.append(
        f"Task A (old capability): `{task_a}`. Task B (new task): `{task_b}`. "
        f"Seed {run.seed}. Every number below is read from `{run.root.name}/checkpoints.csv`, "
        f"`eval/m0_m1.json` and `summary.json`."
    )
    lines.append("")

    # --- Stage B ---
    if m0m1:
        m0, m1, v = m0m1["m0"], m0m1["m1"], m0m1["validity"]
        gain = v["old_task_gain"]
        verdict = "Yes." if v["stage_b_valid"] else "**No — stage B is invalid.**"
        trans = v["old_task_transitions_m0_to_m1"]
        evidence = (
            f"- {task_a}: M0 {_pct(m0['old']['accuracy'])} → M1 {_pct(m1['old']['accuracy'])} "
            f"(gain {100 * gain:+.1f} points; threshold {100 * v['min_old_task_gain']:.0f}). "
            f"Parse failures M0 {_pct(m0['old']['parse_failure_rate'])}, M1 {_pct(m1['old']['parse_failure_rate'])}.\n"
            f"- Per-example: {trans['wrong_to_correct']} wrong→correct, {trans['correct_to_wrong']} correct→wrong "
            f"of n={m0['old']['n']}.\n"
            f"- {task_b} at M0 {_pct(m0['new']['accuracy'])} → M1 {_pct(m1['new']['accuracy'])} "
            f"(side effect of Task-A training on Task B)."
        )
        for key in m0:
            if key.startswith("control_"):
                evidence += f"\n- Control {key[8:]}: M0 {_pct(m0[key]['accuracy'])} → M1 {_pct(m1[key]['accuracy'])}."
        lines.append(_answer("Did Stage B learn Task A?", verdict, evidence))
    else:
        lines.append(
            _answer(
                "Did Stage B learn Task A?",
                "Not run.",
                "`eval_m0_m1` has not been executed for this run.",
            )
        )

    # --- Stage C per arm ---
    for arm, s in arms.items():
        label = ARM_LABEL.get(arm, arm)
        learned = s["new_acc_end"] > s["new_acc_start"]
        lines.append(
            _answer(
                f"Did Stage C ({label}) learn Task B?",
                "Yes." if learned else "**No.**",
                f"- {task_b}: {_pct(s['new_acc_start'])} at step 0 (M1) → {_pct(s['new_acc_end'])} at step {s['steps'][-1]}; "
                f"peak {_pct(s['new_acc_max'])} at step {s['new_acc_max_step']}.",
            )
        )
        forgot = s["forgetting_max"] > 0
        lines.append(
            _answer(
                f"Did Stage C ({label}) forget Task A?",
                "Yes." if forgot else "**No forgetting observed.**",
                f"- {task_a}: {_pct(s['old_acc_start'])} at M1 → {_pct(s['old_acc_end'])} at the final checkpoint "
                f"(forgetting {100 * (s['forgetting_end'] or 0):+.1f} points; worst {_pct(s['old_acc_min'])} at step {s['old_acc_min_step']}).\n"
                f"- Per-example at the final checkpoint: {s['correct_to_wrong_end']} correct→wrong, "
                f"{s['wrong_to_correct_end']} wrong→correct.",
            )
        )
        lines.append(
            _answer(
                f"Did KL increase ({label})?",
                "Yes." if (s["kl_old_end"] or 0) > 0 or (s["kl_new_end"] or 0) > 0 else "**No.**",
                f"- Final checkpoint: kl_old = {_num(s['kl_old_end'])}, kl_new = {_num(s['kl_new_end'])} nats/token "
                f"(KL(M1 ‖ checkpoint), exact token-level, teacher-forced on M1 continuations).",
            )
        )
        r = s["kl_old_vs_new_r"]
        lines.append(
            _answer(
                f"Did KL_old and KL_new behave differently ({label})?",
                "They are close to collinear."
                if (r is not None and r > 0.98)
                else ("Yes, they separate." if r is not None else "Too few checkpoints to say."),
                f"- Pearson r(kl_old, kl_new) across checkpoints = {_num(r, 3)}; "
                f"r(forgetting, kl_old) = {_num(s['forgetting_vs_kl_old_r'], 3)}, "
                f"r(forgetting, kl_new) = {_num(s['forgetting_vs_kl_new_r'], 3)}.",
            )
        )

    # --- Self-SFT viability ---
    st = summary.get("self_sft_stats")
    if st:
        lines.append(
            _answer(
                "Was Self-SFT constructible?",
                "Yes." if st["n_kept"] >= 0.25 * st["n_prompts"] else "**Marginal / not viable.**",
                f"- {st['n_prompts']} prompts × {st['samples_per_prompt']} samples at T={st['temperature']}: "
                f"{st['n_correct_samples']} correct samples ({_pct(st['sample_accuracy'])}), "
                f"{st['n_prompts_solved']} prompts solved ({_pct(st['prompt_solve_rate'])}), {st['n_kept']} kept.",
            )
        )

    # --- Checkpoint table ---
    lines += [
        "## Checkpoint table",
        "",
        "| arm | step | acc_A | acc_B | forgetting | kl_old | kl_new | c→w | w→c | loss |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(rows, key=lambda r: (r["arm"], r["step"])):
        lines.append(
            f"| {r['arm']} | {r['step']} | {_pct(r['old_acc'])} | {_pct(r['new_acc'])} | {_pct(r['forgetting'])} | "
            f"{_num(r['kl_old'])} | {_num(r['kl_new'])} | {r['correct_to_wrong'] if r['correct_to_wrong'] is not None else '—'} | "
            f"{r['wrong_to_correct'] if r['wrong_to_correct'] is not None else '—'} | {_num(r['train_loss'], 3)} |"
        )
    lines.append("")

    # --- Flags ---
    lines += ["## Unexpected results (automatic checks)", ""]
    if summary.get("flags"):
        lines += [f"- {f}" for f in summary["flags"]]
    else:
        lines.append("- None of the automatic checks fired.")
    lines.append("")

    # --- Figures ---
    if figures:
        lines += ["## Figures", ""]
        for name, paths in figures.items():
            rel = Path(paths["png"]).relative_to(run.root)
            lines.append(f"![{name}]({rel.as_posix()})")
        lines.append("")

    target = run.root / "RESULTS.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
