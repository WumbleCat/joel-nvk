#!/usr/bin/env python
"""Build the Self-SFT dataset from M1's own verified Task-B answers.

Thin alias for `python -m joel_nvk.poc.cli --stages build_self`; all flags pass through
(--env, --run-id, --arms, --seed, --low-memory).
"""

from joel_nvk.poc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(default_stages=["build_self"], description=__doc__))
