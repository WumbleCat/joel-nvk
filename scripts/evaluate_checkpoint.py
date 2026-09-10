#!/usr/bin/env python
"""Evaluate every stage-C checkpoint on Task A and Task B, with KL vs M1 on both axes.

Thin alias for `python -m joel_nvk.poc.cli --stages eval_ckpts`; all flags pass through
(--env, --run-id, --arms, --seed, --low-memory).
"""

from joel_nvk.poc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(default_stages=["eval_ckpts"], description=__doc__))
