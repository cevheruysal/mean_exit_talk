#!/usr/bin/env python3
"""Regenerate every figure of the talk from fresh (or seeded) samples.

    python scripts/make_figures.py                 # book parameters (slow, minutes)
    python scripts/make_figures.py --fast          # smoke test (seconds)
    python scripts/make_figures.py --seed 42       # reproducible
    python scripts/make_figures.py --outdir figs   # custom output directory

Writes PNG (and PDF) files into --outdir and prints the two experiment
tables plus the fitted convergence order q.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meanexit import (TOY, CONV, rng_factory, dt_sweep, grid_sweep,
                      print_table, u_exact, figures)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--seed", type=int, default=None,
                    help="master seed (default: fresh randomness)")
    ap.add_argument("--fast", action="store_true",
                    help="reduced M and dt grid for a quick smoke test")
    ap.add_argument("--formats", default="png,pdf")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    get_rng = rng_factory(args.seed)

    if args.fast:
        M_hist, M_conv = 5_000, 50_000
        Ms_tab = (1_000, 10_000)
        dts_tab = (1e-2, 1e-3)
        dts_conv = (1e-1, 1e-2, 1e-3)
    else:                     # the talk's parameters (Section 4.2)
        M_hist, M_conv = 50_000, 500_000
        Ms_tab = (1_000, 10_000, 100_000)
        dts_tab = (1e-1, 1e-2, 1e-3)
        dts_conv = (1e-1, 1e-2, 1e-3, 1e-4)

    def save(fig, name):
        for ext in formats:
            fig.savefig(out / f"{name}.{ext}", dpi=150)
        print(f"  -> {out / name}.{'/'.join(formats)}")

    t0 = time.time()
    print("[1/7] sampling error (hist + running mean)")
    fig, _ = figures.fig_sampling_error(TOY, get_rng("sampling"), M=M_hist)
    save(fig, "01_sampling_error")

    print("[2/7] discretization error (EM vs exact path)")
    save(figures.fig_discretization_error(TOY, get_rng("discretization")),
         "02_discretization_error")

    print("[3/7] missed exits (Figure 12.1)")
    save(figures.fig_missed_exit(TOY, get_rng("missed")), "03_missed_exits")

    print("[4/7] u(x) (Figure 12.2)")
    save(figures.fig_u(TOY), "04_u_of_x")

    print("[5/7] u(x) + exit-time distribution heatmap with CIs "
          "(the Section 4.1 slide)")
    heat_kw = dict(n_x=20, M=1_000, dt=5e-3) if args.fast else {}
    save(figures.fig_u_with_heatmap(TOY, **heat_kw),
         "05_u_panel")

    print("[6/7] M x dt grid of estimates + CIs  (Section 4.2 experiment)")
    res_grid = grid_sweep(TOY, dts_tab, Ms_tab, get_rng, method="exact",
                          name="grid")
    for row in res_grid:
        print_table(row, exact=float(u_exact(TOY.X0, TOY)),
                    title=f"GBM, exact updates, M={row[0].M}:")
    save(figures.fig_m_dt_grid(TOY, res_grid), "06_m_dt_grid")

    print("[7/7] convergence order (Figure 12.4)")
    res_conv = dt_sweep(CONV, dts_conv, M_conv, get_rng, method="exact",
                        name="convergence")
    print_table(res_conv, exact=float(u_exact(CONV.X0, CONV)),
                title=f"GBM (mu={CONV.mu}, X0={CONV.X0}), M={M_conv}:")
    fig, (q, C, resid) = figures.fig_convergence(CONV, res_conv)
    save(fig, "07_convergence")
    print(f"least-squares fit of (12.5):  q = {q:.4f}  (residual {resid:.4f})")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
