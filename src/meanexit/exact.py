"""Closed-form reference quantities for the toy problem (GBM)."""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from .params import Params


def gbm_exact(
        p: Params, 
        t: NDArray[np.float64], 
        W: NDArray[np.float64]
    ) -> NDArray[np.float64]:
    """Exact solution (5.5):  X(t) = X0 exp((mu - sigma^2/2) t + sigma W(t))."""
    t: NDArray[np.float64] = np.asarray(t, dtype=np.float64)
    W: NDArray[np.float64] = np.asarray(W, dtype=np.float64)
    return p.X0 * np.exp((p.mu - 0.5 * p.sigma**2) * t + p.sigma * W)


def gbm_mean(
        p: Params, 
        t: NDArray[np.float64]
    ) -> NDArray[np.float64]:
    """Expected value (5.6):  E[X(t)] = X0 e^{mu t}."""
    return p.X0 * np.exp(p.mu * np.asarray(t, dtype=np.float64))


def u_exact(
        x: NDArray[np.float64], 
        p: Params
    ) -> NDArray[np.float64]:
    """Mean exit time function u(x) of GBM from (a, b) -- equation (12.4).

    u(x) = 1/(sigma^2/2 - mu) [ log(x/a)
             - (1 - (x/a)^{1-2mu/sigma^2}) / (1 - (b/a)^{1-2mu/sigma^2}) log(b/a) ]
    """
    x: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    half_s2: float = 0.5 * p.sigma**2
    gamma: float = 1.0 - 2.0 * p.mu / p.sigma**2
    if abs(gamma) < 1e-9:
        return np.log(x / p.a) * np.log(p.b / x) / p.sigma**2
    num: NDArray[np.float64] = -np.expm1(gamma * np.log(x / p.a))
    den: NDArray[np.float64] = -np.expm1(gamma * np.log(p.b / p.a))
    return (np.log(x / p.a) - (num / den) * np.log(p.b / p.a)) / (half_s2 - p.mu)


def u_argmax(
        p: Params, 
        n: int = 20001
    ) -> Tuple[float, float]:
    """Location and value of the maximum of u on [a, b].

    For the book's parameters (TOY) the maximum sits near x = 0.75:
    closer to a, because paths drift upward on average, see (5.6).
    """
    x: NDArray[np.float64] = np.linspace(p.a, p.b, n)
    u: NDArray[np.float64] = u_exact(x, p)
    i: int = int(np.argmax(u))
    return x[i], u[i]