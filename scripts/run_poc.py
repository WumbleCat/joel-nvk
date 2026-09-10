#!/usr/bin/env python
"""Run the whole proof-of-concept pipeline.

    uv run scripts/run_poc.py --config configs/poc.yaml --low-memory
    uv run scripts/run_poc.py --env poc_smoke

--config <path> is accepted for the brief's sake and maps to --env <stem>.
"""

import sys
from pathlib import Path

from joel_nvk.poc.cli import main


def _translate(argv: list[str]) -> list[str]:
    out = []
    it = iter(argv)
    for arg in it:
        if arg == "--config":
            out += ["--env", Path(next(it)).stem]
        elif arg.startswith("--config="):
            out += ["--env", Path(arg.split("=", 1)[1]).stem]
        else:
            out.append(arg)
    return out


if __name__ == "__main__":
    raise SystemExit(main(_translate(sys.argv[1:]), description=__doc__))
