#!/usr/bin/env python
"""Probe every candidate task with the untouched base model M0.

Thin alias for `python -m joel_nvk.poc.cli --stages probe`; all flags pass through
(--env, --run-id, --arms, --seed, --low-memory).
"""

from joel_nvk.poc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(default_stages=["probe"], description=__doc__))
