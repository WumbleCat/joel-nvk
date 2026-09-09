---
name: paper-writing
description: How to write up this project — framing the matched-KL contribution, the section-by-section structure, tying every claim to a figure and a number, hedging language calibrated to the evidence, and the limitations that must be stated. Use when drafting or revising any section, an abstract, a figure caption, or a rebuttal.
---

# Paper writing

## The contribution, in one sentence

Prior work compares SFT and on-policy/RL post-training at equal training budget
and concludes RL forgets less; that comparison confounds **how far the model
moved** with **how it was trained**. We control drift explicitly and ask whether
the method effect survives.

Lead with the confound, not with the methods. The design *is* the contribution.

## Structure

**Abstract.** The confound, the matched-KL design, the four arms, the headline
result stated as a direction with a number, and which hypothesis it supports.

**Introduction.** The observation from prior work → why budget-matching is not
drift-matching → the three hypotheses stated crisply → what the paper does →
findings as bullets.

**Related work.** Catastrophic forgetting in post-training; RL vs SFT
comparisons; KL-regularised fine-tuning; on-policy distillation and self-training.
Be explicit and fair about which prior claims this work is re-examining, and about
what those papers did control for.

**Setup.** Base model, new task, old-task suite, the four arms, the KL ladder, the
two KL axes and their estimators, seeds. A reader must be able to see that the
arms differ only in method. Justify `self_sft` and `iter_sft` as the arms that
separate data source from objective.

**Results.** One subsection per hypothesis, in order:
- H1 (drift-magnitude): forgetting vs KL, all arms, both axes.
- H2 (data-source): the residual method effect after conditioning on KL.
- H3 (measurement-axis): `kl_old` vs `kl_new` as predictors, and the change in
  the method gap between the two matchings.

**Analysis / ablations.** Data freshness, temperature, filtering strictness,
adaptation surface. New-task performance alongside every forgetting result.

**Limitations.** Non-negotiable; see below.

**Conclusion.** What a practitioner should now do differently.

## Rules for claims

- Every claim names the **axis**, the **achieved KL**, the **effect size with
  interval**, and the **seed count**. "RL forgets less" is not a sentence this
  paper contains without those attached.
- Every quantitative sentence points at a specific figure or table, and that
  figure was produced by committed code from committed results files.
- Report new-task performance next to every forgetting claim. A method that
  forgets less while learning less has not been shown protective.
- Pre-specified analyses and exploratory ones are labelled as such. Do not present
  a post-hoc slice as if it were planned.
- Do not round an interval that straddles zero into a directional claim.

## Calibrated language

| Evidence | Say |
| --- | --- |
| CI excludes zero, several seeds, replicates across benchmarks | "reliably lower" |
| CI excludes zero on one benchmark only | "lower on X; not on Y" |
| CI includes zero, design powered for the effect | "no reliable difference; we can rule out gaps larger than Δ" |
| CI includes zero, underpowered | "inconclusive at this seed count" |

Never write "proves". A well-powered null supporting H1 is a real result and
should be presented with confidence, not apology — but it is a null, and the
detectable-effect bound must accompany it.

## Figures

The main figure is forgetting vs KL, one line per method, one panel per axis,
individual seeds visible behind the means, log-x. A reader should be able to see
H1 (overlapping lines) or H2 (separated lines) without reading the caption.

Captions are self-contained: what is plotted, on which axis, how many seeds, what
the bands mean, and what the reader should take away.

## Limitations to state plainly

- One base model / model scale — the mechanism may be scale-dependent.
- One new task family; forgetting depends on how far the new task sits from the
  old distribution.
- KL is one drift measure among several; parameter-space and representation-space
  distances may order the methods differently.
- Matching is approximate, bounded by checkpoint granularity and estimator noise;
  state the achieved tolerance.
- Seed counts and the smallest detectable effect.
- Old-task suite coverage: forgetting was measured on these capabilities, not all
  of them.

## Rebuttal habits

Answer the number that was asked about. If a reviewer asks for an arm, a scale or
an ablation the design can support, run it and report it — including when it
weakens the claim. Point to the pre-registered analysis when defending against a
"you fished for this" objection; that is what it is for.
