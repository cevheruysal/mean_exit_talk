"""meanexit -- numerics for the seminar talk on Chapter 12 of
Higham & Kloeden, *An Introduction to the Numerical Simulation of
Stochastic Differential Equations* (SIAM, 2021): Mean Exit Times.

Running example (the "toy problem"): geometric Brownian motion

    dX(t) = mu X(t) dt + sigma X(t) dW(t),        (5.4)

with exit interval (a, b) and deterministic X(0) = X0 in (a, b).
"""

from .params import Params, TOY, CONV, rng_factory
from .exact import u_exact, gbm_exact, gbm_mean, u_argmax
from .bvp import solve_bvp_fd
from .simulate import (
    MCResult,
    exit_times_exact,
    exit_times_em,
    exit_times_em_gbm,
    summarize,
    dt_sweep,
    grid_sweep,
    fit_order,
    print_table,
)

__all__ = [
    "Params", "TOY", "CONV", "rng_factory",
    "u_exact", "gbm_exact", "gbm_mean", "u_argmax",
    "solve_bvp_fd",
    "MCResult", "exit_times_exact", "exit_times_em", "exit_times_em_gbm",
    "summarize", "dt_sweep", "grid_sweep", "fit_order", "print_table",
]
