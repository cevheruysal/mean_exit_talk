"""Monte Carlo computation of mean exit times -- the algorithm of Section 12.3.

Two path engines:

* :func:`exit_times_em`     -- Euler--Maruyama update (8.3), line 7 of the
  pseudocode, for a general SDE (12.1);
* :func:`exit_times_exact_gbm`  -- the *exact* GBM update from (5.5), the
  replacement used in Section 12.4 to remove the discretization error.

Both record, per path, the midpoint exit time  T_exit^s = t_n - dt/2
(line 10 of the pseudocode), where t_n is the first grid point at which
the path is observed outside (a, b).
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .params import Params


@dataclass(frozen=True)
class MCResult:
    """One Monte Carlo run: a_M, b_M and the 95% confidence interval (2.6)."""

    dt: float
    M: int
    mean: float   # a_M   = (1/M) sum_s T_exit^s
    std: float    # b_M,  b_M^2 = (1/(M-1)) sum_s (T_exit^s - a_M)^2

    @property
    def half_width(self) -> float:
        return 1.96 * self.std / np.sqrt(self.M)

    @property
    def ci(self) -> Tuple[float, float]:
        return (self.mean - self.half_width, self.mean + self.half_width)

    def bias(self, exact: float) -> float:
        return self.mean - exact

    def row(self, exact: Optional[float] = None) -> str:
        lo, hi = self.ci
        s: str = f"{self.dt:8.0e}  {self.mean:8.4f}  [{lo:7.4f}, {hi:7.4f}]"
        if exact is not None:
            s += f"  {self.mean - exact:+8.4f}"
        return s


def summarize(samples: NDArray[np.float64], dt: float) -> MCResult:
    return MCResult(
        dt=dt,
        M=samples.size,
        mean=float(np.mean(samples)),
        std=float(np.std(samples, ddof=1)),
    )


def exit_times_exact(
    p: Params,
    dt: float,
    M: int,
    rng: np.random.Generator,
    block_budget: int = 2**22,
    max_steps: Optional[int] = None,
) -> NDArray[np.float64]:
    """M exit-time samples for GBM, paths advanced by the exact update

        X_{n+1} = X_n exp((mu - sigma^2/2) dt + sqrt(dt) xi_n sigma),   (from (5.5))

    so the values at the grid points carry *no discretization error*:
    the only remaining errors are sampling and missed exits.

    Vectorized over paths and, in log space, over blocks of steps
    (the log-price is a random walk with iid Gaussian increments).
    Memory use is bounded by ``block_budget`` doubles per array.
    """
    drift: float = (p.mu - 0.5 * p.sigma**2) * dt
    sd: float = p.sigma * np.sqrt(dt)
    la: float = np.log(p.a)
    lb: float = np.log(p.b)

    y: NDArray[np.float64] = np.full(M, np.log(p.X0))          # log-pos of alive paths
    steps: NDArray[np.int64] = np.zeros(M, dtype=np.int64)     # steps taken so far
    idx: NDArray[np.int64] = np.arange(M)                      # original indices of alive paths
    exit_step: NDArray[np.int64] = np.zeros(M, dtype=np.int64)

    while idx.size:
        K: int = int(max(1, min(block_budget // idx.size, 65536)))
        Z: NDArray[np.float64] = rng.standard_normal((idx.size, K))
        Z *= sd
        Z += drift
        Y: NDArray[np.float64] = np.cumsum(Z, axis=1) + y[:, None]

        out: NDArray[np.bool_] = (Y <= la) | (Y >= lb)
        hit: NDArray[np.bool_] = out.any(axis=1)
        first: NDArray[np.int64] = np.argmax(out, axis=1)      # first exit within block

        e: NDArray[np.int64] = np.nonzero(hit)[0]
        exit_step[idx[e]] = steps[e] + first[e] + 1

        s: NDArray[np.int64] = np.nonzero(~hit)[0]
        y = Y[s, -1]
        steps = steps[s] + K
        idx = idx[s]
        if max_steps is not None and idx.size and steps[0] > max_steps:
            raise RuntimeError("max_steps exceeded -- does the SDE exit (a,b)?")

    return (exit_step - 0.5) * dt             # line 10:  t_n - dt/2


def exit_times_em(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    g: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    X0: float,
    a: float,
    b: float,
    dt: float,
    M: int,
    rng: np.random.Generator,
    max_steps: int = 10**8,
) -> NDArray[np.float64]:
    """M exit-time samples with the EM update, line 7 of the pseudocode:

        X_{n+1} = X_n + dt f(X_n) + sqrt(dt) xi_n g(X_n).           (8.3)

    Vectorized over the alive paths; general drift f and diffusion g.
    """
    sqdt: float = np.sqrt(dt)
    X: NDArray[np.float64] = np.full(M, float(X0))
    idx: NDArray[np.int64] = np.arange(M)
    exit_step: NDArray[np.int64] = np.zeros(M, dtype=np.int64)
    n: int = 0
    while idx.size:
        n += 1
        if n > max_steps:
            raise RuntimeError("max_steps exceeded -- does the SDE exit (a,b)?")
        xi: NDArray[np.float64] = rng.standard_normal(idx.size)
        X = X + dt * f(X) + sqdt * xi * g(X)
        out: NDArray[np.bool_] = (X <= a) | (X >= b)
        if out.any():
            e: NDArray[np.int64] = np.nonzero(out)[0]
            exit_step[idx[e]] = n
            keep: NDArray[np.bool_] = ~out
            X = X[keep]
            idx = idx[keep]
    return (exit_step - 0.5) * dt


def exit_times_em_gbm(
    p: Params, dt: float, M: int, rng: np.random.Generator, **kw
) -> NDArray[np.float64]:
    """EM engine specialized to the toy problem (5.4)."""
    return exit_times_em(lambda x: p.mu * x, lambda x: p.sigma * x, p.X0, p.a, p.b, dt, M, rng, **kw)


# ----------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------

def dt_sweep(
    p: Params,
    dts: List[float],
    M: int,
    get_rng: Callable[[str], np.random.Generator],
    method: str = "exact",
    name: str = "sweep",
) -> List[MCResult]:
    """Run the Monte Carlo algorithm for several stepsizes.

    ``get_rng``: the factory from :func:`meanexit.rng_factory` (each dt
    gets its own independent stream so runs are exchangeable).
    """
    results: List[MCResult] = []
    for dt in dts:
        rng = get_rng(f"{name}-dt={dt!r}")
        if method == "exact":
            T: NDArray[np.float64] = exit_times_exact(p, dt, M, rng)
        elif method == "em":
            T = exit_times_em_gbm(p, dt, M, rng)
        else:
            raise ValueError(f"Unknown method: {method}")
        results.append(summarize(T, dt))
    return results


def grid_sweep(
    p: Params,
    dts: List[float],
    Ms: List[int],
    get_rng: Callable[[str], np.random.Generator],
    method: str = "exact",
    name: str = "grid",
) -> List[List[MCResult]]:
    """Run the Monte Carlo algorithm on the full M x dt grid.

    ``result[i][j]`` used M = Ms[i] and dt = dts[j]; each row is an
    ordinary dt sweep, so it feeds :func:`print_table` and
    :func:`figures.fig_convergence` unchanged.
    """
    return [
        dt_sweep(p, dts, M, get_rng, method=method, name=f"{name}-M={M}")
        for M in Ms
    ]


def fit_order(dts: List[float], errors: List[float]) -> Tuple[float, float, float]:
    """Least-squares fit of the power law (12.5):  err ~ C dt^q.

    Fit  log err = log C + q log dt  (cf. Figure 8.2);  returns
    (q, C, residual) with residual = || log err - fit ||_2.
    """
    L: NDArray[np.float64] = np.log(np.asarray(dts, dtype=np.float64))
    E: NDArray[np.float64] = np.log(np.asarray(errors, dtype=np.float64))
    A: NDArray[np.float64] = np.vstack([np.ones_like(L), L]).T
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    logC, q = coef
    resid: float = float(np.linalg.norm(A @ coef - E))
    return float(q), float(np.exp(logC)), resid


def print_table(
    results: List[MCResult], exact: Optional[float] = None, title: str = ""
) -> str:
    """Book-style summary table of a dt sweep."""
    lines: List[str] = []
    if title:
        lines.append(title)
    head = f"{'dt':>8}  {'a_M':>8}  {'95% CI':>18}"
    if exact is not None:
        head += f"  {'bias':>8}"
    lines.append(head)
    lines.append("-" * len(head))
    for r in results:
        lines.append(r.row(exact))
    if exact is not None:
        lines.append(f"exact mean exit time (12.4): {exact:.4f}")
    out = "\n".join(lines)
    print(out)
    return out