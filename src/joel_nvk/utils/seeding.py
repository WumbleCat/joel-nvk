"""Seeding.

One integer seed per run drives training; generation gets its own derived seed so
that sampling does not shift when unrelated RNG consumers change.
"""

import os
import random

MAX_SEED = 2**31 - 1


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch (CPU + CUDA) from a single run seed."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is in the ml group; pure-logic tests run without it
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def derive_seed(seed: int, *parts: object) -> int:
    """Derive a stable sub-seed from the run seed and a label.

    Used for generation so that, e.g., the KL probe always samples the same way
    regardless of how much training RNG was consumed before it.
    """
    key = "|".join([str(seed), *(str(p) for p in parts)])
    # zlib.crc32 is stable across runs and platforms, unlike hash().
    import zlib

    return zlib.crc32(key.encode("utf-8")) % MAX_SEED
