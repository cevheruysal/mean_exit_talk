import numpy as np
import matplotlib.pyplot as plt

from .params import Params, rng_factory
from .exact import gbm_exact, u_exact, u_argmax
from .simulate import (exit_times_em_gbm, exit_times_exact_gbm,
                       summarize, fit_order)

# --- figure sizes (inches) ---
PLOT_HEIGHT = 3.82
PLOT_WIDTH = 6.18
PLOT_WIDTH_LONG = 9.54

# --- text sizes (also serve as small/medium/large annotation tiers) ---
FS_TITLE = 11                # figure / single-panel titles
FS_PANEL = 10                # subplot titles inside a multi-panel figure
FS_LEGEND = 9                # legends and small annotations

# --- palette: one semantic colour per role in the talk ---
CLR_RED     = "tab:red"      # the exact / true answer (u(X0), boundaries of truth)
CLR_GREEN   = "tab:green"    # pitfall solutions
CLR_BLUE    = "tab:blue"     # our computed quantities: MC estimates and analytic u(x)
CLR_ORANGE  = "tab:orange"   # coarse stepsize  dt_1
CLR_CYAN    = "tab:cyan"     # fine stepsize  dt_2
CLR_GRAY    = "0.60"         # a continuous sample path
CLR_FAINT   = "0.40"
CLR_WHITE   = "w"            # markers drawn over the heatmap
CLR_M_RAMP  = ("#4f9dd1", "#2b7bba", "#08519c")

# --- reusable kwarg bundles (extend the EXACT_KW / BARRIER_KW pattern) ---
RED_DASHED      = dict(ls="--", color=CLR_RED,   lw=1.4)    # dashed red line
GREEN_DASHED    = dict(ls="--", color=CLR_GREEN, lw=1.4)    # dashed green line
BLUE_SOLID      = dict(ls="-",  color=CLR_BLUE,  lw=1.0)    # solid blue line
GRAY_SOLID      = dict(ls="-",  color=CLR_FAINT, lw=0.8)    # solid gray line
GRAY_DASHED     = dict(ls="--", color=CLR_FAINT, lw=0.8)    # dashed gray line
GRAY_DOTTED     = dict(ls=":",  color=CLR_FAINT, lw=1.0)    # dotted gray line

GRAY_SHADE = dict(color=CLR_FAINT, alpha=0.75)              # gray shading
BLUE_SHADE = dict(color=CLR_BLUE, alpha=0.35)                 # blue shading

FAINT_KW    = dict(alpha=0.15, ms=2)                     # per-step error dots
HEATMEAN_KW = dict(fmt="x", color=CLR_WHITE, ms=4.5,   # MC means over the heatmap
                   mew=1.2, capsize=2, lw=0.9)
GRIDPT_KW   = dict(fmt="o", ms=5, capsize=3, lw=1.4)     # M x dt markers (colour per cluster)

# --- readable numeric defaults ---
THOUSAND = 1_000
TEN_THOUSAND = 10_000
HUNDRED_THOUSAND = 100_000
ONE_TENTH = 1e-1
ONE_HUNDREDTH = 1e-2
ONE_THOUSANDTH = 1e-3
ONE_TEN_THOUSANDTH = 1e-4

# module-level generator for figures that manage their own randomness
get_rng = rng_factory(None)


# ======================================================================
# private helpers
# ======================================================================

def _exit_times(p: Params, dt: float, M: int, rng, method: str = "em"):
    """M exit-time samples via the EM engine (``"em"``) or the exact-update
    engine (``"exact"``, discretization error switched off)."""
    engine = exit_times_em_gbm if method == "em" else exit_times_exact_gbm
    return engine(p, dt, M, rng)


def _running_stats(T):
    """Running sample mean and 95% CI half-width as functions of the number
    of paths m = 1..M (Monte Carlo (2.6))."""
    M = T.size
    m = np.arange(1, M + 1)
    csum, csumsq = np.cumsum(T), np.cumsum(T * T)
    rmean = csum / m
    with np.errstate(invalid="ignore", divide="ignore"):
        rvar = (csumsq - m * rmean**2) / np.maximum(m - 1, 1)
    rhw = 1.96 * np.sqrt(np.maximum(rvar, 0.0) / m)
    return m, rmean, rhw


def _draw_running_mean(ax, m, rmean, rhw, lo=30):
    """Draw the running sample mean with its shrinking 95% CI funnel."""
    ax.fill_between(m[lo:], (rmean - rhw)[lo:], (rmean + rhw)[lo:], 
                    **BLUE_SHADE,
                    label="95% CI")
    ax.plot(m[lo:], rmean[lo:], 
            **BLUE_SOLID, 
            label="sample mean $a_m$")
    ax.set_xscale("log")
    ax.set_xlim(lo, m[-1])
    ax.set_xlabel("number of paths $m$")


def _em_path(p: Params, W, t_f, stride: int):
    """One Euler-Maruyama trajectory (8.3) on the sub-grid t_f[::stride],
    driven by the fine Brownian path W."""
    t = t_f[::stride]
    dt = t[1] - t[0]
    X = np.empty(t.size)
    X[0] = p.X0
    for n in range(t.size - 1):
        dWn = W[(n + 1) * stride] - W[n * stride]
        X[n + 1] = X[n] + dt * p.mu * X[n] + p.sigma * X[n] * dWn
    return t, X


# ======================================================================
# Section 2: errors encountered in MC simulation of the mean exit time
# ======================================================================

def fig_sampling_error(p: Params, rng,
                       M: int = HUNDRED_THOUSAND, dt: float = ONE_TENTH,
                       method: str = "em"):
    """Error source 1 (sampling): histogram of the exit times (the book's
    Figure 12.3) and the running sample mean with its shrinking 95% CI.

    The CI funnel narrows like O(m^{-1/2}) -- but watch where it heads:
    not exactly to u(X0).  That gap is the theme of Section 4.2.
    """
    T = _exit_times(p, dt, M, rng, method)
    m, rmean, rhw = _running_stats(T)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PLOT_WIDTH_LONG, PLOT_HEIGHT),
                                   constrained_layout=True)

    # --- ax1: histogram of the first 1000 exit times, with the CI band ---
    M1 = min(M, THOUSAND)
    res = summarize(T[:M1], dt)
    se = res.std / np.sqrt(M1)
    ci_low, ci_high = res.mean - 1.96 * se, res.mean + 1.96 * se

    ax1.hist(T[:M1], bins=60, 
             **GRAY_SHADE,
             label=f"histogram of $m={M1}$ exit times")
    ax1.axvline(res.mean, 
                **BLUE_SOLID, 
                label=f"$a_m={res.mean:.4f}$")
    ax1.axvspan(ci_low, ci_high,
                **BLUE_SHADE,  
                label=f"$a_m \\pm \\frac{{b_m}}{{\\sqrt{{m}}}}, "
                      f"\\quad b_m={res.std:.2f}$")
    
    ax1.set_xlim(0, 35)
    ax1.set_xlabel(r"$T^{s}_{exit}$ for sample paths")
    ax1.legend(loc="upper right", 
               fontsize=FS_LEGEND)
    # ax1.set_title(f"histogram of the $m = {M1}$ exit times", 
    #               fontsize=FS_PANEL)

    # --- ax2: running mean with the CI funnel ---
    _draw_running_mean(ax2, m, rmean, rhw)
    ax2.axhline(rmean[-1], 
                **GREEN_DASHED,
                label=f"$a_M={rmean[-1]:.4f}$")
    
    ax2.legend(loc="upper right", 
               fontsize=FS_LEGEND)
    # ax2.set_title(r"sampling error shrinks like $O(m^{-1/2})$", 
    #               fontsize=FS_PANEL)

    fig.suptitle(f"Error source 1: Monte Carlo sampling    "
                 f"(MC-EM,  $\\Delta t={dt:g}$)", 
                 fontsize=FS_TITLE)
    return fig, res


def fig_discretization_error(p: Params, rng, T: float = 16.0,
                             dt_coarse: float = 1.0, refine: int = 4096):
    """Error source 2 (discretization): one Brownian path driving the exact
    GBM solution (5.5) and two EM approximations with stepsizes dt_coarse
    and dt_coarse/4 (left); the running mean of the absolute EM error,
    roughly halved along with the stepsize (right).
    """
    factor = 4
    n_f = int(round(T / dt_coarse)) * refine
    dt_f = dt_coarse / refine
    W = np.r_[0.0, np.cumsum(rng.standard_normal(n_f) * np.sqrt(dt_f))]
    t_f = np.linspace(0.0, T, n_f + 1)
    X_f = gbm_exact(p, t_f, W)

    t1, X1 = _em_path(p, W, t_f, refine)
    t2, X2 = _em_path(p, W, t_f, refine // factor)

    # absolute error at the EM nodes (t = 0 excluded, error there is 0)
    abs1 = np.abs(X1[1:] - X_f[::refine][1:])
    abs2 = np.abs(X2[1:] - X_f[::refine // factor][1:])
    cumavg1 = np.cumsum(abs1) / np.arange(1, abs1.size + 1)
    cumavg2 = np.cumsum(abs2) / np.arange(1, abs2.size + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PLOT_WIDTH_LONG, PLOT_HEIGHT),
                                   constrained_layout=True)

    # --- ax1: the paths ---
    ax1.plot(t_f, X_f, 
             **GRAY_DASHED,
             label="exact solution X(T)")
    ax1.plot(t1, X1, 
             "o-", color=CLR_ORANGE, lw=1.0, ms=4,
             label=f"EM, $\\Delta t_1={dt_coarse:g}$")
    ax1.plot(t2, X2, 
             ".-", color=CLR_CYAN, lw=0.8, ms=3,
             label=f"EM, $\\Delta t_2={dt_coarse/factor:g}$")
    
    ax1.set_xlabel("$t$")
    ax1.legend(fontsize=FS_LEGEND)

    # --- ax2: cumulative mean absolute error, with per-step error behind ---
    ax2.plot(t1[1:], cumavg1, 
             "--", color=CLR_ORANGE, lw=1.2,
             label=f"$\\Delta t_1={dt_coarse:g}$")
    ax2.plot(t2[1:], cumavg2, 
             "--", color=CLR_CYAN, lw=1.2,
             label=f"$\\Delta t_2={dt_coarse/factor:g}$")
    ax2.plot(t1[1:], abs1, 
             "o", color=CLR_ORANGE, **FAINT_KW)
    ax2.plot(t2[1:], abs2, 
             ".", color=CLR_CYAN, **FAINT_KW)

    ax2.set_xlabel("$t$")
    ax2.set_ylabel("mean $|X^{EM} - X|$")
    ax2.legend(fontsize=FS_LEGEND)
    ax2.set_title("cumulative mean EM error", 
                  fontsize=FS_PANEL)

    fig.suptitle("Error source 2: numerical discretization    "
                 f"(EM,  $\\Delta t_1={dt_coarse:g}, "
                 f"\\Delta t_2={dt_coarse/factor:g}$)", fontsize=FS_TITLE)
    return fig


def fig_missed_exit(p: Params, rng, dt: float = 0.618, refine: int = 2048,
                    max_tries: int = 800, T_max: float = 64.0):
    """Recreation of Figure 12.1: a path (solid) sneaks outside (a, b)
    between grid points without being spotted by the discrete samples
    (circles); only a later excursion is detected.

    Draws fresh paths until one with at least one missed excursion before
    the detected exit is found (the most instructive of ``max_tries``
    candidates is kept).
    """
    n_f = int(round(T_max / dt)) * refine
    dt_f = dt / refine
    t_f = np.linspace(0.0, T_max, n_f + 1)

    # --- find an instructive path: exiting, with missed excursions before ---
    best, best_missed = None, -1
    for _ in range(max_tries):
        W = np.r_[0.0, np.cumsum(rng.standard_normal(n_f) * np.sqrt(dt_f))]
        X_f = gbm_exact(p, t_f, W)
        X_c = X_f[::refine]
        out = (X_c <= p.a) | (X_c >= p.b)
        if not out[1:].any():
            continue
        k = 1 + int(np.argmax(out[1:]))              # first detected exit node
        spans = [j for j in range(k - 1)             # earlier missed excursions
                 if X_f[j * refine:(j + 1) * refine + 1].max() >= p.b
                 or X_f[j * refine:(j + 1) * refine + 1].min() <= p.a]
        if len(spans) > best_missed:
            best, best_missed = (X_f, k, spans), len(spans)
        if best_missed >= 2:
            break
    if best is None:
        raise RuntimeError("no exiting path found; increase T_max")
    X_f, k, spans = best

    # --- plot: continuous path + discrete samples + barriers ---
    n_show = min(k * refine + refine // 2, n_f)
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH_LONG, PLOT_HEIGHT),
                           constrained_layout=True)
    ax.plot(t_f[:n_show + 1], X_f[:n_show + 1], 
            **GRAY_SOLID,
            label="solution path X(t)")
    tc = np.arange(k + 1) * dt
    ax.plot(tc, X_f[::refine][:k + 1], 
            "o", color=CLR_BLUE, ms=6,
            label=r"samples $X(t_n)$, $\Delta t=%g$" % dt)
    ax.axhline(p.b, 
               **GRAY_DOTTED)
    ax.axhline(p.a, 
               **GRAY_DOTTED)
    for level, name in ((p.a, "$a$"), (p.b, "$b$")):
        ax.annotate(name, xy=(0.005, level), xycoords=("axes fraction", "data"),
                    ha="left", va="bottom", fontsize=FS_TITLE)

    # --- annotate the missed excursions, then the detected exit ---
    offset_over = offset_under = 0.0
    for j in spans:
        seg = X_f[j * refine:(j + 1) * refine + 1]
        over, under = seg >= p.b, seg <= p.a
        i = int(np.argmax(over | under))             # first crossing in segment
        x, y = t_f[j * refine + i], seg[i]
        y_text = (y + 0.10 + offset_over) if y >= p.b else (y - 0.10 - offset_under)
        offset_over += 0.05 if over.any() else 0.0
        offset_under += 0.05 if under.any() else 0.0
        ax.annotate("missed", xy=(x, y), xytext=(x, y_text),
                    ha="right", fontsize=FS_LEGEND, color=CLR_RED,
                    arrowprops=dict(arrowstyle="->", color=CLR_RED))

    x, y = tc[k], X_f[::refine][k]
    y_text = (y + 0.10 + offset_over) if y >= p.b else (y - 0.10 - offset_under)
    ax.annotate("detected", xy=(x, y), xytext=(x, y_text),
                ha="right", fontsize=FS_LEGEND, color=CLR_GREEN,
                arrowprops=dict(arrowstyle="->", color=CLR_GREEN))

    ax.set_xlabel("$t$")
    ax.set_ylim(p.a - 0.40 - offset_under, p.b + 0.40 + offset_over)
    ax.legend(loc="upper left", fontsize=FS_LEGEND)
    ax.set_title("Error source 3: missed exits between grid points    "
                 "(Fig 12.1)", fontsize=FS_TITLE)
    return fig


def fig_bias_to_exact(p: Params, rng,
                      M: int = HUNDRED_THOUSAND, dt: float = ONE_TENTH,
                      method: str = "em"):
    """The reveal: the running sample mean and its 95% CI, with the exact
    u(X0) drawn in.  The funnel converges (in m) to its own centre --
    which sits *beside* the exact line, not on it: precision, not accuracy.
    """
    T = _exit_times(p, dt, M, rng, method)
    m, rmean, rhw = _running_stats(T)
    exact = float(u_exact(p.X0, p))

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH_LONG, PLOT_HEIGHT),
                           constrained_layout=True)
    _draw_running_mean(ax, m, rmean, rhw)
    ax.axhline(exact, 
               **RED_DASHED, 
               label=f"$u(X_0={p.X0:g}) = {exact:.4f}$")
    ax.legend(loc="upper right", fontsize=FS_LEGEND)

    fig.suptitle(f"Error source 1: Monte Carlo sampling    "
                 f"(MC-EM,  $\\Delta t={dt:g}$)", fontsize=FS_TITLE)
    return fig


# ======================================================================
# Section 4.1: the mean exit time function u(x), equation (12.4)
# ======================================================================

def _ax_u(ax, p: Params):
    """Draw the closed-form u(x) from (12.4) on ax, with X0 and the
    interior maximum annotated."""
    x = np.linspace(p.a, p.b, 800)
    xs, us = u_argmax(p)
    u0 = float(u_exact(p.X0, p))

    ax.plot(x, u_exact(x, p), 
            **RED_DASHED, 
            label="closed form (12.4)")
    ax.plot([p.X0], [u0], 
            "o", color=CLR_RED, zorder=5)
    ax.annotate(f"$u(X_0={p.X0:g}) = {u0:.4f}$",
                xy=(p.X0, u0), xytext=(p.X0 + 0.08, u0 + 0.35), fontsize=FS_PANEL,
                arrowprops=dict(arrowstyle="->", color=CLR_RED), color=CLR_RED)
    ax.axvline(xs, 
               **GRAY_DOTTED)
    ax.annotate(f"max at $x^\\ast\\approx{xs:.2f}$\n(closer to $a$: drift pushes up)",
                xy=(xs, us), xytext=(xs + 0.05, us - 6.0), fontsize=FS_LEGEND,
                arrowprops=dict(arrowstyle="->", color=CLR_GRAY), color=CLR_FAINT)
    
    ax.set_xlabel("initial data, $x$")
    ax.set_xlim(p.a, p.b)
    ax.set_ylabel("$u(x)$")
    ax.set_ylim(0.0, 10.0)
    ax.legend(fontsize=FS_LEGEND)
    ax.set_title("analytic solution $u(x)$", fontsize=FS_TITLE)


def _ax_u_heatmap(ax, p: Params, n_x: int = 40,
                  M: int = HUNDRED_THOUSAND, dt: float = ONE_TENTH,
                  n_t: int = 90, q_max: float = 0.99):
    """Draw, on ax, one column per starting value x0: the column-normalized
    histogram of exit times (bright ridge = mode), the per-column MC means
    with 95% CIs, and the closed-form u(x)."""
    xs = np.linspace(p.a, p.b, n_x + 2)[1:-1]
    samples = [exit_times_em_gbm(p.replace(X0=float(x)), dt, M, get_rng(f"u-heatmap-{x}"))
               for x in xs]
    means = np.array([s.mean() for s in samples])
    hw = 1.96 * np.array([s.std(ddof=1) for s in samples]) / np.sqrt(M)

    # bin edges snapped to the dt-lattice so the two grids do not alias
    t_max = float(np.quantile(np.concatenate(samples), q_max))
    width = max(1, int(round(t_max / (n_t * dt)))) * dt
    t_edges = np.arange(0.0, t_max + width, width)
    H = np.stack([np.histogram(s, bins=t_edges)[0] for s in samples], axis=1).astype(float)
    H /= H.max(axis=0)                               # column-normalize: ridge = mode

    col = (p.b - p.a) / (n_x + 1)                    # column width
    x_edges = np.r_[xs - col / 2, xs[-1] + col / 2]
    x_fine = np.linspace(p.a, p.b, 512)

    ax.pcolormesh(x_edges, t_edges, H, cmap="viridis", rasterized=True)
    ax.plot(x_fine, u_exact(x_fine, p), 
            **RED_DASHED, 
            label="closed form (12.4)")
    ax.errorbar(xs, means, yerr=hw, 
                **HEATMEAN_KW, 
                label="MC means $\\pm$ 95% CI")
    
    ax.set_xlabel("initial data, $x$")
    ax.set_xlim(p.a, p.b)
    ax.set_ylabel(r"$T_{exit}$")
    ax.set_ylim(0.0, 10.0)
    ax.legend(fontsize=FS_LEGEND, loc="upper right")
    ax.set_title(r"exit-time heatmap    (MC-EM, M=%g, $\Delta t=%g$)" % (M, dt),
                 fontsize=FS_TITLE)


def fig_u(p: Params):
    """The mean exit time function u(x) from the closed form (12.4)
    (the book's Figure 12.2)."""
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT), constrained_layout=True)
    _ax_u(ax, p)
    return fig


def fig_u_heatmap(p: Params, **kw):
    """The exit-time distribution heatmap with MC means (see _ax_u_heatmap)."""
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT), constrained_layout=True)
    _ax_u_heatmap(ax, p, **kw)
    return fig


def fig_u_with_heatmap(p: Params, **heat_kw):
    """The Section 4.1 slide: the analytic u(x) (left, Fig. 12.2) beside its
    Monte Carlo approximation and the full exit-time distribution
    (right, Fig. 12.5)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PLOT_WIDTH_LONG, PLOT_HEIGHT),
                                   constrained_layout=True)
    _ax_u(ax1, p)
    _ax_u_heatmap(ax2, p, **heat_kw)
    fig.suptitle("Toy problem: mean exit time    (Figs 12.2 - 12.5)", fontsize=FS_TITLE)
    return fig


# ======================================================================
# Section 4.2: the M x dt grid, and the empirical order of convergence
# ======================================================================

def _pow10_label(M: int) -> str:
    e = np.log10(M)
    return f"$M=10^{{{int(round(e))}}}$" if np.isclose(e, round(e)) else f"$M={M}$"


def fig_m_dt_grid(p: Params, grid):
    """Sample means and 95% CIs on the full M x dt grid, one dodged cluster
    per stepsize: within a cluster growing M only tightens the interval
    around the same biased centre, while shrinking dt moves the whole
    cluster down onto u(X0) -- the two knobs are orthogonal.  Open markers
    flag CIs that miss the exact value.
    """
    exact = float(u_exact(p.X0, p))
    n = len(grid)

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT), constrained_layout=True)

    handles = []
    for i, row in enumerate(grid):
        color = CLR_M_RAMP[i % len(CLR_M_RAMP)]
        dodge = 1.25 ** (i - (n - 1) / 2)            # multiplicative on a log x-axis
        for r in row:
            lo, hi = r.ci
            covers = lo <= exact <= hi
            ax.errorbar(r.dt * dodge, r.mean, yerr=r.half_width, color=color,
                        markerfacecolor=color if covers else "white", 
                        **GRIDPT_KW)
        handles.append(plt.Line2D([], [], marker="o", ms=5, lw=1.4,
                                  color=color, label=_pow10_label(row[0].M)))
    ax.axhline(exact, 
               **RED_DASHED)
    ax.annotate(f"exact $u(X_0) = {exact:.4f}$", xy=(0.99, exact - 0.03),
                xycoords=("axes fraction", "data"), ha="right", va="top",
                color=CLR_RED, fontsize=FS_PANEL)
    
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel("mean exit time estimate")
    ax.legend(handles=handles, fontsize=FS_LEGEND, loc="upper left")
    ax.set_title(r"$CI\, width \propto M$,   $Bias \propto \Delta t$    "
                 r"(MC w/ exact-update)", fontsize=FS_TITLE)
    return fig


def fig_convergence(p: Params, results):
    """The log-log error plot with a slope-1/2 reference and the
    least-squares fit of (12.5) (right panel of the book's Figure 12.4;
    the CIs themselves live in :func:`fig_m_dt_grid`).
    """
    exact = float(u_exact(p.X0, p))
    dts = np.array([r.dt for r in results])
    errs = np.abs(np.array([r.mean for r in results]) - exact)
    q, C, resid = fit_order(dts, errs)

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT), constrained_layout=True)
    
    ax.loglog(dts, errs, 
              "x", ms=9, color=CLR_BLUE, 
              label="observed error")
    ax.loglog(dts, C * dts**q, 
              **BLUE_SOLID,
              label=f"LS fit: $q={q:.2f}$ (resid {resid:.4f})")
    ref = errs[-1] * 10 * (dts / dts[-1]) ** 0.5
    ax.loglog(dts, ref, 
              **GRAY_DASHED, 
              label="reference slope $1/2$")
    
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$|a_M - u(X_0)|$")
    ax.legend(fontsize=FS_LEGEND, loc="upper left")
    ax.set_title(f"Empirical error convergence    "
                 f"(MC w/ exact-update, $M={results[0].M}$)", fontsize=FS_TITLE)
    return fig, (q, C, resid)
