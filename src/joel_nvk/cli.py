"""Command line entry point: ``joel-nvk <command>``."""

import argparse
import logging
from collections.abc import Sequence

from joel_nvk import __version__
from joel_nvk.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="joel-nvk", description="joel-nvk command line interface"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--env", default="default", help="Config in configs/ to load (default: default)"
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("config", help="Print the resolved configuration")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    config = load_config(args.env)
    if args.command == "config":
        for key, value in config.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
