"""ess.pdf, the paper's figure: the ESS of a description against r, the
fraction of prior variance it leaves, with n = 0, 1, 10, 100 demonstrations in
hand, one panel per reliability p = 1, 0.99, 0.9; d = 16, a0 = 10. The dotted
line is the limit of the n = 0 ESS as r -> 0 (the r = 0 rows of ess_curve.csv).

ess_curve.py computes ess_curve.csv; this script only draws it. The figure style
comes from the description-icl project (results/rq2-reliability).

    python3 docs/paper/figs/ess_figure.py
"""

import csv
import pathlib
import sys

here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here.parents[3] / "description-icl" / "results" / "rq2-reliability"))
import orx_figstyle as fs  # noqa: E402

RS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
PS = (1.0, 0.99, 0.9)
NS = (0, 1, 10, 100)

rows = [{k: float(v) for k, v in x.items()} for x in csv.DictReader((here / "ess_curve.csv").open())]
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
    ax.set_yticks([0.1, 1, 10, 100]); ax.set_yticklabels(["0.1", "1", "10", "100"]); ax.set_ylim(0.07, 400)
    ax.set_xlabel(r"$r$ (more specific $\rightarrow$)")
    ax.text(0.04, 0.95, f"$p = {p:g}$", transform=ax.transAxes, va="top")
    ax.grid(True, axis="y", color=fs.MUTED, lw=0.5)
for ax in axes:
    ax.set_ylabel("ESS (demonstrations)")
fs.panel_labels(axes)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False)
fs.save(fig, str(here / "ess"))
