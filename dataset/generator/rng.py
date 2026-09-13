"""Deterministic random-number streams.

Every random decision is drawn from a stream derived from the master seed and
a *name*, never from global state or from the order in which samples happen
to be processed.  Consequences:

* sample ``i`` depends only on the configuration, the master seed and ``i``,
  so the same configuration and seed always give the same batch;
* each effect draws from its own sub-stream, so enabling one effect does not
  shift the random values every later effect sees;
* samples can later be generated in parallel without changing the output.

Seeds are derived with SHA-256 rather than Python's ``hash()``, which is
salted per process.
"""

from __future__ import annotations

import hashlib

import numpy as np

_SEED_BYTES = 8


def derive_seed(master_seed: int, *keys: object) -> int:
    """Return a 64-bit seed derived from ``master_seed`` and ``keys``."""
    text = ":".join([str(int(master_seed)), *(str(key) for key in keys)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:_SEED_BYTES], "little")


def make_rng(master_seed: int, *keys: object) -> np.random.Generator:
    """A NumPy generator for the stream named by ``keys``."""
    return np.random.Generator(np.random.PCG64(derive_seed(master_seed, *keys)))


class SampleRng:
    """Named, independent random streams for one sample."""

    def __init__(self, sample_seed: int) -> None:
        self.sample_seed = int(sample_seed)

    def stream(self, name: str) -> np.random.Generator:
        return make_rng(self.sample_seed, name)
