#!/usr/bin/env python
"""Stage C: M1 -> M2 for the chosen arms, snapshotting adapters at save_steps.

Thin alias for `python -m joel_nvk.poc.cli --stages train_new`; all flags pass through
(--env, --run-id, --arms, --seed, --low-memory).
"""

from joel_nvk.poc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(default_stages=["train_new"], description=__doc__))
