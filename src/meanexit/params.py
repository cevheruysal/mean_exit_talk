"""Parameter sets and randomness management."""

from dataclasses import dataclass, replace as _replace
import zlib

import numpy as np


@dataclass(frozen=True)
class Params:
    """Parameters of the toy problem: GBM (5.4) on the interval (a, b).

    dX = mu X dt + sigma X dW,  X(0) = X0,  0 < a < X0 < b.
    """

    mu: float
    sigma: float
    a: float
    b: float
    X0: float

    def replace(self, **kw) -> "Params":
        return _replace(self, **kw)

    def __post_init__(self):
        if not (0.0 < self.a < self.X0 < self.b):
            raise ValueError("need 0 < a < X0 < b")


TOY = Params(mu=0.1, sigma=0.2, a=0.5, b=2.0, X0=1.0)

CONV = Params(mu=0.5, sigma=0.2, a=0.5, b=2.0, X0=1.5)


def rng_factory(master_seed=None):
    """Return ``get_rng(name)`` handing out one Generator per experiment.

    * ``master_seed=None``  -> fresh, independent randomness on *every*
      call: re-running a notebook cell recomputes its figure from new
      samples (live-demo mode).
    * ``master_seed=<int>`` -> reproducible: the generator for a given
      ``name`` is always the same, independent across names.
    """

    def get_rng(name: str) -> np.random.Generator:
        if master_seed is None:
            return np.random.default_rng()
        tag = zlib.crc32(str(name).encode("utf-8"))  # stable across runs
        return np.random.default_rng(np.random.SeedSequence([master_seed, tag]))

    return get_rng
