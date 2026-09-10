#!/usr/bin/env python
"""KL vs M1 on old-task and new-task prompts for every stage-C checkpoint.

KL is computed together with checkpoint accuracy (one model load per
checkpoint), so this is an alias for the ``eval_ckpts`` stage. All flags pass
through (--env, --run-id, --arms, --seed, --low-memory).
"""

from joel_nvk.poc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(default_stages=["eval_ckpts"], description=__doc__))
