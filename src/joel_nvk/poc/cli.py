"""Command line for the PoC pipeline; every script in ``scripts/`` is a thin alias.

python -m joel_nvk.poc.cli --env poc                    # everything
python -m joel_nvk.poc.cli --env poc --stages probe     # one stage
python -m joel_nvk.poc.cli --env poc --stages train_new eval_ckpts --arms self_sft --run-id <id>
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from joel_nvk.poc.pipeline import ARMS, STAGES, make_run_id, run_poc
from joel_nvk.utils import load_config, outputs_dir, setup_logging
from joel_nvk.utils.console import use_utf8_output
from joel_nvk.utils.net import use_system_certs

logger = logging.getLogger("poc")


def build_parser(
    default_stages: Sequence[str] | None = None, description: str | None = None
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description or __doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--env", default="poc", help="Config in configs/ (default: poc; poc_smoke for a dry run)"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGES),
        default=list(default_stages) if default_stages else None,
        help="Stages to run, in pipeline order (default: all)",
    )
    parser.add_argument(
        "--arms", nargs="+", choices=list(ARMS), help="Stage-C arms (default: from config)"
    )
    parser.add_argument("--run-id", help="Resume an existing run")
    parser.add_argument("--seed", type=int, help="Override the config seed")
    parser.add_argument("--low-memory", action="store_true", help="One process per stage")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    default_stages: Sequence[str] | None = None,
    description: str | None = None,
) -> int:
    use_utf8_output()
    use_system_certs()
    args = build_parser(default_stages, description).parse_args(argv)
    config = load_config(args.env)
    if args.seed is not None:
        config["project"]["seed"] = args.seed

    run_id = args.run_id or make_run_id(config)
    setup_logging(config["logging"]["level"], log_file=outputs_dir() / "logs" / f"{run_id}.log")

    run = run_poc(
        config,
        stages=args.stages,
        arms=args.arms,
        run_id=run_id,
        low_memory=args.low_memory,
        env_name=args.env,
    )
    print(f"\nrun_id: {run.run_id}\nroot:   {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
