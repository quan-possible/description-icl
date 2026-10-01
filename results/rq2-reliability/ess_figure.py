"""ess.pdf, the paper's figure of the ESS of a description against its
precision, with n = 0, 1, 10, 100 demonstrations in hand, one panel per
reliability p = 1, 0.99, 0.9; d = 16, a0 = 10, a dense log-spaced grid of r,
one seed of 8,000 prompts. The dotted line is the limit of the n = 0 ESS as
r -> 0, where the identification term is all of H(p); in ess_curve.csv the
r = 0 rows hold that limit and the r = -1 rows the lower-bound cap, the ESS at
which R_ex(n) = (1-p) R_ex(0). Computes ess_curve.csv first and reuses it if present.

    uv run python results/rq2-reliability/ess_figure.py
"""

import csv
import pathlib

import orx_figstyle as fs

from description_icl import gaussian as g
from description_icl import mixture as mx

here = pathlib.Path(__file__).parent
D, A0, K, S = 16, 10.0, 260, 8000
RS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
PS = (1.0, 0.99, 0.9)
NS = (0, 1, 10, 100)

out = here / "ess_curve.csv"
if not out.exists():
    plain = mx.simulate(D, A0, A0 * 0.999, 1.0, K, S, seed=100)
    R_ex = mx.regret(plain, 0.0, 1)
    R_cap = mx.regret(mx.simulate(D, A0, A0 * 0.999, 1.0, 400, S // 4, seed=101), 0.0, 1)  # longer grid: the p = 0.99 floor cap is near 330
    rows = [dict(p=p, r=-1, n=0, worth=round(float(g.first_crossing(R_cap, (1 - p) * R_cap[0])[1]), 3)) for p in PS if p < 1]  # floor cap
    for p in PS:
        if p < 1:  # the limit as r -> 0: the description pins w exactly or is wrong
            R_lim = mx.regret(mx.simulate(D, A0, 1e-6 * A0, p, K, S, seed=200), p, 1)
            rows.append(dict(p=p, r=0, n=0, worth=round(float(g.first_crossing(R_ex, R_lim[0])[1]), 3)))
    for p in PS:
        for r in RS:
            R_desc = mx.regret(mx.simulate(D, A0, r * A0, p, K, S, seed=200), p, 1)
            for n in NS:
                rows.append(dict(p=p, r=r, n=n, worth=round(float(g.first_crossing(R_ex, R_desc[n])[1]) - n, 3)))
            print(f"p={p} r={r} done", flush=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
rows = [{k: float(v) for k, v in x.items()} for x in csv.DictReader(out.open())]
curve = lambda p, n: [next(x["worth"] for x in rows if x["p"] == p and x["n"] == n and x["r"] == r) for r in RS]
cap = lambda p: next(x["worth"] for x in rows if x["p"] == p and x["r"] == 0)

fs.use_style()
fig, axes = fs.figure_grid(1, 3, width=fs.TEXT, ratio=0.36)
NCOL = dict(zip(NS, fs.family("blue", 4)))
for ax, p in zip(axes, PS):
    for n in NS:
        ax.plot(RS, curve(p, n), "-o", ms=2.5, color=NCOL[n], lw=1.3, label=f"$n = {n}$")
    if p < 1:
        ax.axhline(cap(p), color=fs.BASELINE, lw=1, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xticks([0.5, 0.1, 0.01, 0.001]); ax.set_xticklabels(["0.5", "0.1", "0.01", "0.001"]); ax.minorticks_off()
    ax.set_yticks([1, 3, 10, 30, 100]); ax.set_yticklabels(["1", "3", "10", "30", "100"]); ax.set_ylim(0.3, 400)
    ax.set_xlabel("precision $r$")
    ax.text(0.04, 0.95, f"$p = {p:g}$", transform=ax.transAxes, va="top")
    ax.grid(True, axis="y", color=fs.MUTED, lw=0.5)
for ax in axes:
    ax.set_ylabel("ESS (demonstrations)")
fs.panel_labels(axes)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False)
fs.save(fig, str(here / "ess"))
