#!/usr/bin/env python
"""Stage B: M0 -> M1 on Task A, then evaluate M0 and M1 and check validity.

Alias for ``--stages prepare train_old eval_m0_m1``; all flags pass through
(--env, --run-id, --arms, --seed, --low-memory).
"""

from joel_nvk.poc.cli import main

if __name__ == "__main__":
    raise SystemExit(
        main(default_stages=["prepare", "train_old", "eval_m0_m1"], description=__doc__)
    )
