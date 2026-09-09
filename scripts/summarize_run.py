#!/usr/bin/env python
"""Print the accuracy, forgetting and KL tables for a finished run.

uv run scripts/summarize_run.py pipeline-Qwen2.5-0.5B-Instruct-s42-20260909-1830
uv run scripts/summarize_run.py --latest
"""

import argparse

from joel_nvk.core.report import format_summary, load_records, summarise
from joel_nvk.utils.console import use_utf8_output
from joel_nvk.utils.net import use_system_certs
from joel_nvk.utils.paths import outputs_dir


def latest_run_id() -> str:
    candidates = sorted(
        (p for p in (outputs_dir() / "results").glob("*.jsonl") if ".predictions" not in p.name),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit("No runs found under outputs/results/")
    return candidates[-1].stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_id", nargs="?", help="Run id to summarise")
    parser.add_argument("--latest", action="store_true", help="Summarise the most recent run")
    return parser.parse_args()


def main() -> int:
    use_utf8_output()
    use_system_certs()
    args = parse_args()
    if not args.run_id and not args.latest:
        raise SystemExit("Give a run_id or pass --latest")

    run_id = args.run_id or latest_run_id()
    print(format_summary(run_id, summarise(load_records(run_id))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
