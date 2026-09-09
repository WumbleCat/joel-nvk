#!/usr/bin/env python
"""Run the base -> LoRA(MATH) -> LoRA(MMLU) pipeline and measure forgetting.

    uv run scripts/run_pipeline.py --env smoke            # does it work at all
    uv run scripts/run_pipeline.py --env pilot            # a first real pass
    uv run scripts/run_pipeline.py --env smoke --stages kl --run-id <existing>

Stages: prepare, baseline, train_math, eval_math, train_mmlu, eval_mmlu, kl.
"""

import argparse
import logging

from joel_nvk.core.pipeline import STAGES, run_pipeline
from joel_nvk.utils import load_config, outputs_dir, setup_logging
from joel_nvk.utils.console import use_utf8_output

logger = logging.getLogger("run_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--env", default="smoke", help="Config in configs/ to load (default: smoke)"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGES),
        help="Subset of stages to run (default: all, in order)",
    )
    parser.add_argument(
        "--run-id",
        help="Reuse an existing run id — resumes from its adapters and appends to its results",
    )
    parser.add_argument("--seed", type=int, help="Override the config seed")
    return parser.parse_args()


def main() -> int:
    use_utf8_output()
    args = parse_args()
    config = load_config(args.env)
    if args.seed is not None:
        config["project"]["seed"] = args.seed

    setup_logging(
        config["logging"]["level"],
        log_file=outputs_dir() / "logs" / f"pipeline-{args.env}.log",
    )

    run = run_pipeline(config, stages=args.stages, run_id=args.run_id)
    print(f"\nrun_id: {run.run_id}")
    print(f"results: {run.results_file}")
    print(f"summarise with: uv run scripts/summarize_run.py {run.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
