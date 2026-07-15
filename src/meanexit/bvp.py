"""Numerical reference solution: finite differences for the BVP (12.2).

    (1/2) g(x)^2 u''(x) + f(x) u'(x) = -1   on (a, b),   u(a) = u(b) = 0.

This is the deterministic route to the mean exit time (the book uses
MATLAB's bvp4c for the same purpose).  Central differences on a uniform
grid give a tridiagonal system, solved here with the Thomas algorithm --
no SciPy required.
"""

from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray
# from scipy.integrate import solve_bvp


# def solve_bvp_scipy(
#     f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
#     g: Callable[[NDArray[np.float64]], NDArray[np.float64]],
#     a: float,
#     b: float,
#     n: int = 2000,
# ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
#     """
#     Solve the BVP using scipy.integrate.solve_bvp (bvp4c equivalent).
#     Returns x (grid), u (solution).
#     """

#     # Vectorised ODE right‑hand side
#     def ode(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
#         u: NDArray[np.float64] = y[0]
#         du: NDArray[np.float64] = y[1]
#         # Guard against zero diffusion (g(x) assumed > 0)
#         ddu: NDArray[np.float64] = (-1.0 - f(x) * du) / (0.5 * g(x) ** 2)
#         return np.vstack((du, ddu))

#     # Residuals for boundary conditions
#     def bc(
#         ya: NDArray[np.float64], yb: NDArray[np.float64]
#     ) -> NDArray[np.float64]:
#         return np.array([ya[0], yb[0]])  # u(a)=0, u(b)=0

#     # Initial mesh and guess – simple linear ramp works
#     x_init: NDArray[np.float64] = np.linspace(a, b, n + 1)
#     y_init: NDArray[np.float64] = np.zeros((2, x_init.size))
#     y_init[0] = np.zeros_like(x_init)
#     y_init[1] = np.ones_like(x_init)

#     sol = solve_bvp(ode, bc, x_init, y_init, tol=1e-8, max_nodes=10000)
#     return sol.x, sol.y[0]


def _thomas(
    lower: NDArray[np.float64],
    diag: NDArray[np.float64],
    upper: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve a tridiagonal system in O(n)."""
    n: int = diag.size
    c: NDArray[np.float64] = np.empty(n)
    d: NDArray[np.float64] = np.empty(n)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom: float = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / denom if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denom
    u: NDArray[np.float64] = np.empty(n)
    u[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        u[i] = d[i] - c[i] * u[i + 1]
    return u


def solve_bvp_fd(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    g: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: float,
    b: float,
    n: int = 2000,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve (12.2) by second-order central finite differences.

    Parameters
    ----------
    f, g : callables (vectorized), drift and diffusion of the SDE (12.1)
    a, b : interval endpoints
    n    : number of subintervals

    Returns
    -------
    x : grid including endpoints, shape (n+1,)
    u : approximation of the mean exit time function on the grid
    """
    x: NDArray[np.float64] = np.linspace(a, b, n + 1)
    h: float = (b - a) / n
    xi: NDArray[np.float64] = x[1:-1]
    alpha: NDArray[np.float64] = 0.5 * g(xi) ** 2 / h ** 2  # multiplies u_{i-1} - 2u_i + u_{i+1}
    beta: NDArray[np.float64] = f(xi) / (2.0 * h)           # multiplies u_{i+1} - u_{i-1}

    lower: NDArray[np.float64] = alpha - beta               # coefficient of u_{i-1}
    diag: NDArray[np.float64] = -2.0 * alpha                # coefficient of u_i
    upper: NDArray[np.float64] = alpha + beta               # coefficient of u_{i+1}
    rhs: NDArray[np.float64] = -np.ones_like(xi)

    inner: NDArray[np.float64] = _thomas(
        np.r_[0.0, lower[1:]], diag, np.r_[upper[:-1], 0.0], rhs
    )
    return x, np.r_[0.0, inner, 0.0]