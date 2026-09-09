#!/usr/bin/env python
"""Fetch the raw datasets this project depends on into ``data/raw/``.

    uv run scripts/download_data.py
"""

import argparse
import logging

from joel_nvk.utils import data_dir, setup_logging

logger = logging.getLogger("download_data")

# name -> URL. Fill in as datasets are added; see data/README.md.
SOURCES: dict[str, str] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    raw = data_dir() / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    if not SOURCES:
        logger.warning("No sources configured — add entries to SOURCES in %s", __file__)
        return 0

    for name, url in SOURCES.items():
        target = raw / name
        if target.exists() and not args.force:
            logger.info("Skipping %s (already present)", name)
            continue
        logger.info("Downloading %s from %s", name, url)
        # TODO: implement the fetch (urllib / requests / cloud SDK).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
