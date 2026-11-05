"""ess.pdf, Figure 2 of the paper: the Bayes-optimal ESS of a description with no demonstrations given,
against its specificity c = 1/(r a0), for a reliable description (p = 1, green) and one
that is relevant with probability 0.9 (pink); d = 16, a0 = 10. Dotted: the limit of the p = 0.9 ESS as
c -> infinity (the r = 0 row of ess_curve.csv). A linear ESS axis shows the cap as a plateau.

Style: paper_style.py (the Editorial Neutral binding). ess_curve.py computes ess_curve.csv; this script only draws it.

    python3 docs/paper/figs/ess_figure.py        # log loss: ess_curve.csv -> ess.pdf
    python3 docs/paper/figs/ess_figure.py sq     # squared error: ess_curve_sq.csv -> ess_sq.pdf
"""

import csv
import pathlib
import sys

import paper_style as ps

here = pathlib.Path(__file__).parent
sq = sys.argv[1:] == ["sq"]
stem = "ess_sq" if sq else "ess"
RS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
CS = [1 / (r * 10) for r in RS]  # specificity c = 1/(r a0), a0 = 10
rows = [{k: float(v) for k, v in x.items()} for x in csv.DictReader((here / ("ess_curve_sq.csv" if sq else "ess_curve.csv")).open())]
curve = lambda p: [next(x["worth"] for x in rows if x["p"] == p and x["n"] == 0 and x["r"] == r) for r in RS]
cap = next(x["worth"] for x in rows if x["p"] == 0.9 and x["r"] == 0)

ps.use_style()
fig, ax = ps.figure(height=2.15)
for p, tone, name in ((1.0, ps.GREEN, "$p = 1$"), (0.9, ps.PINK, "$p = 0.9$")):
    ax.plot(CS, curve(p), color=tone.edge, lw=1.8, marker="o", ms=4, mfc=tone.fill, mec=tone.edge, mew=0.9, label=name)
ax.axhline(cap, color=ps.PINK.edge, lw=1.1, ls=(0, (1, 2)), label="$p = 0.9$ as $c \\to \\infty$")
ax.set_xscale("log")
ax.set_xticks([0.2, 1, 10, 100]); ax.set_xticklabels(["0.2", "1", "10", "100"]); ax.minorticks_off()
ax.set_ylim(0, 125); ax.set_yticks([0, 25, 50, 75, 100, 125])
ax.set_xlabel("specificity $c$ (demonstrations)")
ax.set_ylabel("ESS (demonstrations)")
ps.style_axes(ax, grid_axis="y")
ax.legend(loc="upper left")
ps.save(fig, str(here / stem))
