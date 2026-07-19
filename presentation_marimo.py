import marimo

__generated_with = "0.23.14"
app = marimo.App(
    width="medium",
    app_title="Mean Exit Times",
    layout_file="layouts/presentation_marimo.slides.json",
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # %% 0 - Setup
    # Make `src/meanexit` importable regardless of the launch directory, then
    # pull in the same names the Jupyter notebook uses.  MASTER_SEED = None ->
    # fresh randomness on every re-run (live-demo mode); set an int to freeze.
    import pathlib
    import sys

    _src = pathlib.Path(__file__).resolve().parent / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = 130  # crisp figures (retina substitute)

    from meanexit import (
        TOY,
        CONV,
        rng_factory,
        u_exact,
        grid_sweep,
        figures,
    )

    MASTER_SEED = None
    get_rng = rng_factory(MASTER_SEED)
    return TOY, figures, get_rng, grid_sweep


@app.cell
def _(mo):
    # Live-demo controls.  Each figure slide carries its own "resample" button;
    # clicking it (in `marimo run` or `marimo edit`) re-runs that figure's cell,
    # and because MASTER_SEED is None every re-run draws brand-new samples.
    # A button must live in a different cell from the one that reads it, so they
    # are all defined here and referenced on the individual figure slides.
    def _resample_button(label="↻ resample"):
        return mo.ui.button(value=0, on_click=lambda n: n + 1, label=label)

    b_sampling = _resample_button()
    b_discretization = _resample_button()
    b_missed = _resample_button()
    b_heatmap = _resample_button()
    b_grid = _resample_button("↻ resample grid  (updates the convergence fit too)")
    return b_discretization, b_grid, b_heatmap, b_missed, b_sampling


@app.cell
def _(mo):
    mo.md(
        r"""
# Mean Exit Times

*An Introduction to the Numerical Simulation of Stochastic Differential Equations*

<span style="font-variant: small-caps;">D. J. Higham</span> and <span style="font-variant: small-caps;">P. E. Kloeden</span>,

---

## Outline:
1. **Motivation & Problem Statement**
2. **Monte Carlo - Euler Maruyama**
3. **Analytical Baseline to Mean Exit Times**
4. **Numerical Example Problem**
5. **Outro**


**Presenter:** *A. Cevher Uysal*
**Date:** *16 July 2026*

---
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
## §1 · Motivation & Problem Definition

$$\;\mathrm{d}X(t) = f(X(t))\,\mathrm{d}t + g(X(t))\, \mathrm{d}W(t), \qquad X(0)=X_0\in(a,b)  \tag{12.1}$$

$$\;T_{\mathrm{exit}} := \inf\{t : X(t)=a \text{ or } X(t)=b\}$$

$$\textbf{Goal:}\text{ estimate } \; T^{\mathrm{mean}}_{\mathrm{exit}} = \mathbb{E}[T_{\mathrm{exit}}]$$

---
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
## §2 · Monte Carlo — Euler–Maruyama

Simulate $M$ sample paths with the EM method,
stop each at the first *grid point* outside $(a,b)$, average sample exit times.

```text
 1   choose a stepsize Δt
 2   choose a number of paths M
 3   for s = 1 to M
 4       set t_n = 0 and X_n = X0
 5       while a < X_n < b
 6           compute a N(0,1) sample ξ_n
 7           replace X_n by X_n + Δt f(X_n) + √Δt ξ_n g(X_n)             ← (EM step)
 8           replace t_n by t_n + Δt
 9       end
10       set T_exit^s = t_n − ½ Δt                       (midpoint of the last step)
11   end
12   set a_M  = (1/M) Σ_s T_exit^s
13   set b_M² = (1/(M−1)) Σ_s (T_exit^s − a_M)²
```

$$\text{95\% confidence interval:}  \qquad
\Big[\,a_M - 1.96\,\tfrac{b_M}{\sqrt{M}},\;\; a_M + 1.96\,\tfrac{b_M}{\sqrt{M}}\,\Big]$$

---
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
### §2.1 · Sources of Error for MC-EM

Two sources of error *we have already covered* in previous chapters

1. **Sampling error** — a sample mean is not an expectation.
2. **Discretization error** — an EM path is not an SDE path.
"""
    )
    return


@app.cell
def _(TOY, b_sampling, figures, get_rng, mo):
    # Error source 1 - MC Sampling
    #
    # Experiment: histogram of the M=1000 EM-simulated exit times
    #             and the running sample mean with its shrinking 95% CI
    #             for M=10^5 paths, Δt = 10^-1.
    _fig, _res = figures.fig_sampling_error(TOY, get_rng("sampling"), M=100_000, dt=1e-1)
    mo.vstack([b_sampling, _fig])  # reading b_sampling => click re-runs this cell
    return


@app.cell
def _(TOY, b_discretization, figures, get_rng, mo):
    # Error source 2 - Discretization
    #
    # Experiment: exact solution and EM simulation differences for two stepsizes
    #             for T=64, Δt ∈ { 1, 0.25 }.
    _fig = figures.fig_discretization_error(TOY, get_rng("discretization"))
    mo.vstack([b_discretization, _fig])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
---

### §2.1 · Sources of Error for MC-EM (cont'd)

And a new source of error inherent to *discrete sampling*

3. **Missed exits error** — we record solution values **only at the grid points** $\{t_i\}$:


within $t_i < t < t_{i+1}$
the path may leave $(a,b)$ *and return unnoticed*
"""
    )
    return


@app.cell
def _(TOY, b_missed, figures, get_rng, mo):
    # Error source 3 - Missed Exits
    #
    # Experiment: plot exact solution to find instances with missed exits
    #             for T_max=64, Δt = 0.618.
    _fig = figures.fig_missed_exit(TOY, get_rng("missed-exits"))
    mo.vstack([b_missed, _fig])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
---

## §4 · A Toy Problem

Consider *Geometric Brownian Motion (GBM)*, where we substitute $f(X) = \mu \cdot X(t)$ and $g(X) = \sigma \cdot X(t)$ into (12.1) to yield:

$$\mathrm{d}X(t) = \mu X(t)\,\mathrm{d}t + \sigma X(t)\,\mathrm{d}W(t) \tag{5.4}$$

with **exact solution** and **mean**

$$X(t) = X_0\,e^{(\mu-\frac{1}{2}\sigma^2)t + \sigma W(t)}
\qquad\qquad
\mathbb{E}[X(t)] = X_0\,e^{\mu t} \tag{5.5 - 5.6}$$

---
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
### §4.1 · $u(x)$ for the toy problem

With $f(x)=\mu \cdot x$ and $g(x)=\sigma \cdot x$, the boundary value problem $\mathcal{A}u=-1$:

$$\tfrac{1}{2}\sigma^2x^2\,u'' + \mu x\,u' = -1
\quad\text{ for } a<x<b, \qquad u(a)=u(b)=0 \tag{12.3}$$

which has the solution

$$u(x) = \frac{1}{\frac{1}{2}\sigma^2-\mu}
\left(\log\frac{x}{a} - \frac{1-(x/a)^{1-2\mu/\sigma^2}}{1-(b/a)^{1-2\mu/\sigma^2}}\,\log\frac{b}{a}\right) \tag{12.4}$$

Our **reference solution** — obtained both *analytically* (12.4) and *numerically*:
"""
    )
    return


@app.cell
def _(TOY, b_heatmap, figures, mo):
    # Mean Exit Time function u(x)
    #
    # Experiment: closed form (12.4) of the derived u(x) and the numerical
    #             approximation with MC-EM on a grid of 40 starting values x0;
    #             for each x0, M=10^4 paths, Δt = 10^-1.
    _fig = figures.fig_u_with_heatmap(TOY)
    mo.vstack([b_heatmap, _fig])  # left panel is closed-form; right panel resamples
    return


@app.cell
def _(mo):
    mo.md(
        r"""
---

### §4.2 · MC-EM for the toy problem — with exact path updates

For GBM we know the exact solution (5.5), so in **line 7** of the pseudocode, instead of

$$X_n \;\leftarrow\; X_n + \Delta t\,\mu X_n + \sqrt{\Delta t}\,\xi_n\,\sigma X_n
\qquad\qquad\text{(EM step (8.3))}$$

we use

$$X_n \;\leftarrow\; X_n \exp\!\Big(\big(\mu-\tfrac{1}{2}\sigma^2\big)\Delta t + \sqrt{\Delta t}\,\xi_n\,\sigma\Big)
\qquad\text{(exact update, from (5.5))}$$

The grid values now carry **no discretization error**: error source 2 is gone,
and whatever error remains is **sampling (E1) + missed exits (E3)**.

**Experiment**: the full grid $M \in \{10^{3},10^{4},10^{5}\} \times \Delta t \in \{10^{-1},10^{-2},10^{-3}\}$.
"""
    )
    return


@app.cell
def _(TOY, b_grid, figures, get_rng, grid_sweep, mo):
    # CI for varying Δt and M
    #
    # Experiment: sample mean and 95% CI on the full M × Δt grid,
    #             M ∈ { 10^3, 10^4, 10^5 },  Δt ∈ { 10^-1, 10^-2, 10^-3 }.
    b_grid  # click -> re-run; res42 changes, so the convergence fit updates too
    res42 = grid_sweep(
        TOY,
        dts=(1e-1, 1e-2, 1e-3),
        Ms=(1000, 10_000, 100_000),
        get_rng=get_rng,
        name="grid",
    )
    _fig = figures.fig_m_dt_grid(TOY, res42)
    mo.vstack([b_grid, _fig])
    return (res42,)


@app.cell
def _(mo):
    mo.md(
        r"""
Reading the grid — the two knobs are **orthogonal**:

1. at fixed $\Delta t$ (one cluster), growing $M$ only **tightens** the interval around the same centre;
2. at $\Delta t = 10^{-3}$ the exact answer lies **well outside** the tightest confidence interval (open markers), and
3. the Monte Carlo method **overestimates** — always from above, exactly as the missed-exit picture predicts.

The CI is **honest about what it measures**: the mean exit time of the discretely observed process
$\{t_i, X(t_i)\}$ — a different, larger number than the one we want.

$$\boxed{\text{The confidence interval measures precision, not accuracy.}}$$

- To improve estimation, **decrease $\Delta t$** — increasing $M$ would only sharpen the wrong target.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
---

### §4.2 · MC-EM for the toy problem — empirical error convergence

$$\mathrm{err}_{\Delta t} := \big|\,a_M - T^{\mathrm{mean}}_{\mathrm{exit}}\big| \approx C\,\Delta t^{\,q} \tag{12.5}$$

and fit $\log \mathrm{err}_{\Delta t} = \log C + q \log \Delta t$ by **least squares** in log–log coordinates.


The book finds $q = 0.45$ (residual $0.1171$) — consistent with the widely reported
$O(\Delta t^{1/2})$ behaviour.

**Experiment**: fit (12.5) to the $M=10^5$ row of the grid above — no new simulation.
"""
    )
    return


@app.cell
def _(TOY, figures, res42):
    # Empirical error convergence rate
    #
    # Experiment: least-squares fit of (12.5) on the M = 10^5 row of the
    #             M × Δt grid from the previous experiment.
    #             for M = 10^5,  Δt ∈ { 10^-1, 10^-2, 10^-3 }.
    _res43 = res42[-1]
    _fig, (_q, _Cfit, _resid) = figures.fig_convergence(TOY, _res43)
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
---

## §5 · Outro &nbsp; **What we did**

1. **Problem**: mean exit time
   1. $T^{\mathrm{mean}}_{\mathrm{exit}} := \mathbb{E}\left[\inf\{t: X(t)=a \;\text{or}\; X(t)=b\}\right]$.
2. **Method**: MC-EM — and its three error sources:
   1. *sampling*,
   2. *discretization*,
   3. *missed exits*.
3. **Baseline**: an analytical dual of our problem
   1. $\;\mathcal{A}u = -1, \,\, u(a)=u(b)=0$ — a deterministic reference.
4. **Toy problem**:
   1. known closed form (12.4);
   2. exact updates isolate the missed-exit error;
   3. the CIs are *precise but biased*;
   4. empirically $\mathrm{err} = O(\Delta t^{1/2})$.

---
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
## §5 · Outro &nbsp; **Questions to keep pondering**

* A natural question from this point: the **probability that the process reaches $a$ before $b$** ?

* The order of convergence $O(\Delta t^{1/2})$ is **very bad** — how can it be improved ?
    * adaptive $\Delta t$ near the boundary
    * after each step, compute the probability that an exit was missed and draw a uniform to decide
    * random exponential stepsizes

* A **rigorous proof** of the error order $O(\Delta t^{1/2})$ ?
    * *(§12.5: not known in this generality — paths may take arbitrarily long to exit, so finite-time convergence theory does not apply)*

* **Other methods** to compute mean exit times ?
    * random-walk-based methods
    * multilevel Monte Carlo
    * apply a numerical method to the deterministic ODE (12.2)
"""
    )
    return


if __name__ == "__main__":
    app.run()
