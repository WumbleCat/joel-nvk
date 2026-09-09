#!/usr/bin/env python
"""Run one experiment and write its artefacts under ``outputs/``.

    uv run scripts/run_experiment.py --env dev --name baseline
"""

import argparse
import json
import logging
from datetime import datetime

from joel_nvk.utils import load_config, outputs_dir, setup_logging

logger = logging.getLogger("run_experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="dev", help="Config in configs/ to load")
    parser.add_argument("--name", default="run", help="Experiment name, used in output paths")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.env)

    run_id = f"{args.name}-{datetime.now():%Y%m%d-%H%M%S}"
    setup_logging(config["logging"]["level"], log_file=outputs_dir() / "logs" / f"{run_id}.log")
    logger.info("Starting %s with config %s", run_id, args.env)

    # TODO: replace with the actual experiment.
    metrics = {"run_id": run_id, "env": args.env, "status": "not-implemented"}

    results = outputs_dir() / "results" / f"{run_id}.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Wrote %s", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
