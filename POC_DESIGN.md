# Proof-of-concept design: does learning a new task move a model away from an old capability, and how do we measure it?

Status: design locked for the PoC run; the benchmark pair is provisional until the
probe stage confirms it (see *Decision log*, D4). Every decision below records
its evidence, the alternatives, and the known limitation.

## 1. Research question

The larger project asks: **if two post-training methods move a model equally far
from its starting point, do they still forget different amounts?** Prior work
compares SFT and on-policy methods at equal training budget and finds on-policy
training forgets less; that confounds *how far the model moved* with *how it was
trained*, and measures "how far" only on the new task's prompts.

The PoC does not answer that. It establishes, on one base model and one task
pair, that we can **observe** the four quantities the design needs — old-task
learning, new-task learning, old-task forgetting, and drift — at checkpoint
resolution, with KL measured separately on old-task and new-task prompts.

## 2. Setup: M0 → M1 → M2

```
M0   Qwen/Qwen2.5-1.5B-Instruct, untouched
 │   stage B: LoRA on Task A, then merged into the weights
M1   old-capability reference checkpoint
 │   stage C: post-training on Task B — one arm at a time, from the same M1
M2   adapter snapshots at fixed optimiser steps
```

- **M1 is the reference** for both forgetting and KL. It is the model whose
  acquired capability we are trying to preserve. M0 → M1 and M0 → M2 are
  diagnostics only.
- Stage-B LoRA is **merged** into the weights so that stage-C adapters attach to
  M1 as a plain base. Disabling a stage-C adapter recovers M1 exactly; that is
  what makes the KL reference free (one set of weights in memory).
- Stage C starts every arm from the identical M1 checkpoint. The arms differ only
  in *where the training targets come from*.

## 3. Benchmark candidates

Base-model numbers are from the Qwen2.5 technical report (Yang et al. 2024,
arXiv:2412.15115, Table 10 for the instruct model; Table 5 / the release blog for
the base model). Reported instruct-model settings are Qwen's own evaluation
harness, not our 0-shot generative prompt, so they set expectations rather than
predict our probe.

| Dataset | Capability | Train split | Eval split | Auto verifier | Reported Qwen2.5-1.5B | Contamination / validity concern | Train? | Eval? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARC-Challenge | grade-school science knowledge, 4-way MC | 1,119 | test 1,172 (val 299) | letter match | 54.7 (base, few-shot) | widely used; inside MMLU auxiliary_train; Qwen reports n-gram decontamination | yes | yes |
| ARC-Easy | same source, easier | 2,251 | test 2,376 | letter match | — | as above | augmentation only | no |
| CommonsenseQA | commonsense over ConceptNet, 5-way MC | 9,741 | validation 1,221 (test unlabeled) | letter match | — | known annotation artefacts (Talmor et al. 2019 §6) | yes | yes |
| Winogrande (xl) | pronoun resolution, binary | 40,398 | validation 1,267 | letter match | 65.0 (base) | binary → chance 50%, high variance; adversarially filtered | yes | weak |
| HellaSwag | sentence completion, 4-way | 39,905 | validation 10,042 | letter match | 67.9 (base) | Chizhov et al. 2025 (arXiv:2504.07825): ~40% ungrammatical prompts, 65% of predictions unchanged with the question removed; "should not be used for evaluation" | no | **rejected** |
| MMLU | 57-subject academic knowledge, 4-way | none native (auxiliary_train = ARC/RACE/OBQA/MC-TEST) | test 14,042 | letter match | 50.7 (MMLU-redux, instruct) | ~6.5% erroneous items (Gema et al. 2024, arXiv:2406.04127); no train split | no | control only |
| GSM8K | grade-school arithmetic, free-form CoT | 7,473 | test 1,319 | exact numeric | 73.2 (instruct) | GSM1k (Zhang et al. 2024, arXiv:2405.00332): up to 8-point drops in some families, frontier models fine; ~1.7% label noise | yes | yes |
| MATH | competition math, free-form | 7,500 | test 5,000 | boxed match | 55.2 (instruct) | heavily represented in Qwen2.5-Math pretraining data | yes | yes |
| MBPP | Python synthesis | 374 | test 500 | unit-test execution | 63.2 (instruct) | needs a sandbox; tiny train split | deferred | deferred |
| HumanEval | Python synthesis | none | 164 | execution | 61.6 (instruct) | no train split | no | control (deferred) |
| IFEval | verifiable instruction following | none | 541 | 25 rule checkers | 42.5 (instruct) | best "capability" candidate; needs the checker suite and a training set (Tulu-3 IF personas, ~30k) | deferred | deferred |
| PIQA | physical commonsense, binary | 16k | validation 1,838 | letter match | — | binary; HF loader is script-based | no | no |

Excluded from the PoC on infrastructure grounds, not merit: MBPP/HumanEval
(sandboxed execution), IFEval (checker suite + constructed training set). Both
are strong candidates for the H100 phase and are listed in *Section 12*.

## 4. Pre-training probe (M0, 0-shot, our prompt, n = 100 per task)

Run `poc-1.5b-0910-1341`, `outputs/poc/poc-1.5b-0910-1341/probe/`. Greedy
decoding; MC answers extracted as a letter, math answers from `\boxed{}`.

| task | acc (M0) | chance | headroom | parse-fail | mean len | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| arc_challenge | 0.750 | 0.25 | 0.25 | 0.00 | 2 | usable; less headroom than expected |
| commonsense_qa | 0.690 | 0.20 | 0.31 | 0.00 | 2 | usable; 9.7k train items |
| winogrande | 0.520 | 0.50 | — | 0.00 | 2 | at chance 0-shot; binary — rejected |
| hellaswag | 0.570 | 0.25 | — | 0.00 | 2 | rejected on validity grounds (D3), probed for the record |
| mmlu | 0.580 | 0.25 | — | 0.00 | 2 | control only (no train split) |
| gsm8k | 0.420 | — | — | **0.44** | 283 | see below |
| math | pending | — | — | — | — | filled in when the probe finishes |

**The GSM8K parse-failure rate was an evaluator problem, not a model problem.**
Inspecting the 44 failures: 42 hit the 320-token generation cap mid-solution
(Qwen2.5-Instruct writes long LaTeX-formatted chains of thought); the 2 that
finished without `\boxed{}` both stated the correct number in prose. Changes made
and recorded (D10): GSM8K budget raised to 512 tokens; a last-number fallback
grades un-boxed prose answers, while `format_failure_rate` and
`truncation_rate` are recorded separately so neither is hidden. Expect M0's
GSM8K accuracy at 512 tokens to land well above 0.42.

## 5. Final pair and justification

**Task A (old capability): CommonsenseQA**, training on 3,000 train items,
evaluated on 300 validation items (test labels are hidden).
**Task B (new task): GSM8K**, training on 2,000 train items, evaluated on 200
test items at 512 tokens.
**Controls: MMLU** (200) and **ARC-Challenge** (200) at M0, M1 and the final
checkpoint — ARC-C is an *untrained* MC capability to set beside the trained one.

```
Recommended Task A: commonsense_qa
Recommended Task B: gsm8k

Why:
  Different capability, different output protocol, different prompt distribution
  (commonsense MC vs. multi-step arithmetic CoT). Both have native train and
  held-out eval splits and exact automatic verifiers. MC evaluation is ≤ 8
  tokens, which is what makes dense checkpoint evaluation affordable on one
  6 GB GPU; GSM8K at 512 tokens is the one expensive evaluation.
  CSQA over ARC-C: lower 0-shot start (0.69 vs 0.75) and 9x the training data,
  so the ≥ 3-point stage-B gain the validity gate demands is the more likely.
Expected old-task learning:
  LoRA on 3k CSQA items: +3 to +8 points 0-shot (format compliance plus the
  dataset's own regularities — Talmor et al. report fine-tuned models well above
  zero-shot ones). Validity gate: ≥ 3 points.
Expected new-task learning:
  Qwen2.5-1.5B-Instruct is strong on GSM8K (73.2 reported by Qwen), so the SFT
  gain on 2k items may be small — the pair's main risk; the Task-B learning flag
  detects it. MATH is the fallback if GSM8K shows no headroom at 512 tokens.
Why forgetting should be measurable:
  GSM8K SFT pushes the output protocol toward long CoT; single-letter answering
  is the kind of narrow protocol that post-training on a different format
  disturbs. format_failure_rate and per-example transitions separate format loss
  from knowledge loss; ARC-C as an untrained control shows whether the loss is
  specific to the trained capability.
Why KL_old vs KL_new may be informative:
  The two prompt distributions are far apart (commonsense MC vs. arithmetic word
  problems), so drift measured on each need not be collinear.
Main confounds:
  (1) Format drift masquerading as forgetting — parse/format/truncation rates and
  transitions. (2) Task-B gains limited by a strong base — Task-B learning flag.
  (3) CSQA annotation artefacts (Talmor et al. §6) — affect absolute level, not the
  within-run M1→M2 comparison. (4) One seed.
```

## 5. Hypotheses for the PoC

- H-B: `acc_A(M1) − acc_A(M0) ≥ 3 points` (stage B creates a measurable capability).
- H-C1: `acc_B(final) > acc_B(M1)` for Vanilla SFT (stage C learns).
- H-C2: `acc_A(step)` decreases with step for at least one arm (forgetting is observable).
- H-KL: `KL_old` and `KL_new` both increase with step, and are not collinear (r < 0.98).
- H-S: Self-SFT retains ≥ 25% of Task-B training prompts (constructible).

## 6. Controls

- Same base model, tokenizer, dtype (bf16), LoRA surface (r=16, α=32, q/k/v/o/gate/up/down) for every arm.
- Same prompt templates, decoding (greedy), extraction and grading code for every checkpoint.
- Frozen, hashed evaluation sets; disjointness from training asserted before anything runs.
- Every arm starts from the identical merged M1; same optimiser, schedule, sequence length, and step ladder.
- MMLU as an untouched control at M0/M1/final.

## 7. Confounds and how they are handled

| Confound | Handling |
| --- | --- |
| Output-format change instead of capability change | Parse-failure rate and mean completion length recorded at every checkpoint; per-example transitions kept |
| Old-task eval items seen in training | `assert_disjoint` on prompts; ARC-Easy augmentation never touches ARC-C test |
| bf16 merge rounding in M1 | Same M1 for every arm, so it cancels in the M1→M2 comparison; documented |
| Self-SFT dataset smaller than SFT's | Same step ladder for both; dataset size reported; `keep_per_prompt=1` keeps one target per question |
| Single seed | Reported as such; the design leaves seeds as a config list for the H100 phase |
| KL estimator noise | Bootstrap SE over prompts recorded with every KL |

## 8. Evaluation metrics

Per checkpoint: `acc_A`, `acc_B`, `forgetting = acc_A(M1) − acc_A(step)`,
parse-failure rate and mean completion length per task, `correct→wrong` and
`wrong→correct` counts against M1's predictions, control accuracy at the final
checkpoint, training loss and learning rate at that step. Per-example
predictions are stored for every evaluation.

## 9. KL methodology

- **Quantity**: exact token-level KL over the full vocabulary, not a sample estimate.
- **Direction (primary)**: `KL(M1 ‖ checkpoint)` — expectation under M1, the model
  whose capability we preserve. This is the direction used by Chen et al. 2025
  ("Retaining by Doing", KL[π₀‖π_θ]). The reverse `KL(checkpoint ‖ M1)` is
  recorded alongside (`kl_*_rev`).
- **Inputs**: 64 prompts per axis. `KL_old` on Task-A evaluation prompts,
  `KL_new` on Task-B evaluation prompts. Never combined.
- **Continuations**: generated **once by M1** (greedy, ≤ 160 new tokens), frozen,
  then teacher-forced through every checkpoint. Every checkpoint is scored on
  identical tokens.
- **Masking**: only continuation positions count; prompt tokens and padding are
  excluded. Logit at position *j−1* scores token *j*.
- **Averaging**: mean over continuation tokens within a prompt, then mean over
  prompts; `n_tokens` and `n_prompts` stored with the value.
- **Precision**: log-softmax in float32 regardless of compute dtype.
- **Uncertainty**: bootstrap standard error over prompts (1,000 resamples).
- **Sanity tests** (`tests/unit/test_kl.py`): KL(model, same model) ≈ 0; KL ≥ 0;
  a hand-computed value; prompt positions never move the number; asymmetry.

## 10. Training arms

| Arm | Targets | From | Notes |
| --- | --- | --- | --- |
| Vanilla SFT | GSM8K reference solutions (calculator annotations stripped, final answer boxed) | M1 | mandatory |
| Self-SFT | M1's own sampled answers (4 per prompt, T=0.7) filtered by the verifier, one kept per prompt | M1 | retention stats reported; flagged if < 25% |
| Iterative-SFT | as Self-SFT, regenerated from the current policy at the start of each of 2 rounds | M1 | opt-in via config |

All arms: LoRA r=16, lr 1e-4 cosine, effective batch 16, 2 epochs over 2,000
prompts = 250 steps; adapters snapshotted at steps 10, 25, 50, 75, 100, 150,
200, 250 (dense early, where drift moves fastest).

## 11. Expected outcomes

- Stage B: +3 to +10 points on ARC-C; MMLU roughly unchanged.
- Vanilla SFT: GSM8K up by a few points then plateau; ARC-C down by a few points,
  most of it in the first 50 steps; KL_new > KL_old at every checkpoint.
- Self-SFT: smaller KL at matched step, less ARC-C loss; GSM8K gain comparable or
  smaller (targets are M1's own solutions).
- The single most useful figure is Plot 4 (forgetting vs KL, both axes): it shows
  whether the two arms trace one curve or two.

## 12. What changes before H100 runs

- ≥ 3 seeds per arm; 1,000+ evaluation items per task.
- Add IFEval as a task (checker suite + Tulu-3 IF training set) and MBPP/HumanEval
  with a sandbox — the capability pairs with the cleanest verifiers.
- Add GRPO as the on-policy RL arm.
- KL ladder matching across arms (the matched-KL analysis proper).
- Repeat with a 7B model; check whether the effect is scale-dependent.

## Decision log

| # | Decision | Evidence / source | Alternatives considered | Reason chosen | Known limitation |
| --- | --- | --- | --- | --- | --- |
| D1 | Base model Qwen2.5-1.5B-Instruct | brief; Qwen2.5 report Table 10; used by Chen et al. 2025 | 0.5B (too weak on GSM8K), 3B (VRAM) | fits 6 GB with LoRA + grad checkpointing; prior work comparability | strong math prior limits Task-B headroom |
| D2 | Merge stage-B LoRA into M1 | PEFT `merge_and_unload`; need M1 as KL reference at zero cost | stack two adapters; keep A unmerged | disabling stage-C adapter recovers M1 exactly | bf16 rounding on merge (shared by all arms) |
| D3 | Reject HellaSwag for evaluation | Chizhov et al. 2025 | keep with GoldenSwag subset | fundamental construct-validity problems | none — probed for the record only |
| D4 | Task A = ARC-C (+ARC-E), Task B = GSM8K (provisional) | dataset cards; Qwen2.5 report; cost model for one 6 GB GPU | CSQA→GSM8K; ARC-C→MATH; GSM8K→ARC-C | different capabilities and protocols, exact verifiers, cheap eval | Task-B headroom; the probe can swap in MATH or CSQA — any swap is recorded here |
| D5 | Primary KL direction KL(M1 ‖ ckpt) | Chen et al. 2025 use KL[π₀‖π_θ] | reverse; symmetric; sampled sequence KL | expectation under the model whose capability we preserve; exact and reproducible | direction choice reorders arms in principle — reverse is recorded too |
| D6 | Teacher-force on M1's continuations | evaluation skill: identical tokens for every checkpoint | each checkpoint's own samples | removes sampling noise from the comparison | continuations are on-policy for M1 only |
| D7 | Greedy decoding for all evaluation | determinism; two evaluations of one checkpoint agree exactly | sampling with n draws | reproducible per-example transitions | greedy can understate a stochastic model |
| D8 | Eight checkpoints, dense early | drift moves fastest at the start | uniform spacing | ladder resolution where KL changes most | eight evaluations per arm on a laptop GPU |
| D9 | Iterative-SFT opt-in | brief §5; complexity budget | always on | PoC value is in SFT vs Self-SFT first | rounds share the step ladder only approximately |
| D10 | **Supersedes D4.** Task A = CommonsenseQA; ARC-C demoted to control; GSM8K budget 320→512 with last-number fallback; b_eval 300→200; six checkpoints [10, 25, 50, 100, 175, 250] | probe run `poc-1.5b-0910-1341` (§4): ARC-C 0.75 vs CSQA 0.69 0-shot; 42/44 GSM8K failures were cap truncations | keep ARC-C; switch Task B to MATH | maximise the chance of a valid stage B; fix the evaluator before it fakes learning; keep total eval time inside one laptop-GPU night | fewer ladder rungs; CSQA artefacts |

## References

- Yang et al. 2024. *Qwen2.5 Technical Report*. arXiv:2412.15115.
- Clark et al. 2018. *Think you have Solved Question Answering? Try ARC*. arXiv:1803.05457.
- Talmor et al. 2019. *CommonsenseQA*. NAACL. arXiv:1811.00937.
- Sakaguchi et al. 2020. *WinoGrande*. AAAI. arXiv:1907.10641.
- Zellers et al. 2019. *HellaSwag*. ACL. arXiv:1905.07830.
- Chizhov et al. 2025. *What the HellaSwag? On the Validity of Common-Sense Reasoning Benchmarks*. arXiv:2504.07825.
- Hendrycks et al. 2021. *Measuring Massive Multitask Language Understanding*. ICLR. arXiv:2009.03300.
- Gema et al. 2024. *Are We Done with MMLU?* (MMLU-Redux). arXiv:2406.04127.
- Cobbe et al. 2021. *Training Verifiers to Solve Math Word Problems* (GSM8K). arXiv:2110.14168.
- Zhang et al. 2024. *A Careful Examination of LLM Performance on Grade School Arithmetic* (GSM1k). arXiv:2405.00332.
- Hendrycks et al. 2021. *Measuring Mathematical Problem Solving With the MATH Dataset*. NeurIPS. arXiv:2103.03874.
- Austin et al. 2021. *Program Synthesis with Large Language Models* (MBPP). arXiv:2108.07732.
- Zhou et al. 2023. *Instruction-Following Evaluation for LLMs* (IFEval). arXiv:2311.07911.
- Chen, Razin, Narasimhan, Chen 2025. *Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting*. arXiv:2510.18874.
- Shenfeld et al. 2025. *RL's Razor: Why Online RL Forgets Less*. arXiv:2509.04259.
