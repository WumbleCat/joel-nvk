#!/usr/bin/env python
"""Aggregate checkpoints.csv, render the figures and write RESULTS.md.

Thin alias for `python -m joel_nvk.poc.cli --stages aggregate`; all flags pass through
(--env, --run-id, --arms, --seed, --low-memory).
"""

from joel_nvk.poc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(default_stages=["aggregate"], description=__doc__))
