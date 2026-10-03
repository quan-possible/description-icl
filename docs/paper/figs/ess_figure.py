"""ess.pdf, Figure 2 of the paper: the Bayes-optimal ESS of a description with no demonstrations in hand,
against r, the fraction of the prior variance it leaves, for a reliable description (p = 1, green) and one
that is relevant with probability 0.9 (pink); d = 16, a0 = 10. Dotted: the limit of the p = 0.9 ESS as
r -> 0 (the r = 0 row of ess_curve.csv). A linear ESS axis shows the cap as a plateau.

Style: paper_style.py (the Editorial Neutral binding). ess_curve.py computes ess_curve.csv; this script only draws it.

    python3 docs/paper/figs/ess_figure.py
"""

import csv
import pathlib

import paper_style as ps

here = pathlib.Path(__file__).parent
RS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
rows = [{k: float(v) for k, v in x.items()} for x in csv.DictReader((here / "ess_curve.csv").open())]
curve = lambda p: [next(x["worth"] for x in rows if x["p"] == p and x["n"] == 0 and x["r"] == r) for r in RS]
cap = next(x["worth"] for x in rows if x["p"] == 0.9 and x["r"] == 0)

ps.use_style()
fig, ax = ps.figure(height=2.15)
for p, tone, name in ((1.0, ps.GREEN, "$p = 1$"), (0.9, ps.PINK, "$p = 0.9$")):
    ax.plot(RS, curve(p), color=tone.edge, lw=1.8, marker="o", ms=4, mfc=tone.fill, mec=tone.edge, mew=0.9, label=name)
ax.axhline(cap, color=ps.PINK.edge, lw=1.1, ls=(0, (1, 2)), label="$p = 0.9$ as $r \\to 0$")
ax.set_xscale("log"); ax.invert_xaxis()
ax.set_xticks([0.5, 0.1, 0.01, 0.001]); ax.set_xticklabels(["0.5", "0.1", "0.01", "0.001"]); ax.minorticks_off()
ax.set_ylim(0, 125); ax.set_yticks([0, 25, 50, 75, 100, 125])
ax.set_xlabel("$r$ (fraction of prior variance left)")
ax.set_ylabel("ESS (demonstrations)")
ps.style_axes(ax, grid_axis="y")
ax.legend(loc="upper left")
ps.save(fig, str(here / "ess"))
