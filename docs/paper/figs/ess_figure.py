"""ess.pdf, Figure 2 of the paper: the Bayes-optimal ESS of a description; d = 16, a0 = 10.

(a) With no demonstrations given, against its precision c = 1/tau^2 (tau: the standard deviation of the task around
the description's mean; a0 = tau_0^2 in the paper's notation), for a reliable description (p = 1, green)
and one that is relevant with probability 0.9 (pink). Dotted: the limit of the p = 0.9 ESS as c -> infinity (the
r = 0 row of ess_curve.csv). A linear ESS axis shows the cap as a plateau.
(b) Squared error only: against the number of demonstrations n, for three c = 100 descriptions (ess_n_sq.csv):
reliable, unreliable (p = 0.9), and a reliable description of half the directions (k = 8).

Style: paper_style.py (the Editorial Neutral binding). ess_curve.py and ess_n_sq.py compute the data; this script only draws it.

    python3 docs/paper/figs/ess_figure.py        # log loss: ess_curve.csv -> ess.pdf, panel (a) only
    python3 docs/paper/figs/ess_figure.py sq     # squared error: ess_curve_sq.csv, ess_n_sq.csv -> ess_sq.pdf
"""

import csv
import pathlib
import sys

import matplotlib.pyplot as plt

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
if sq:
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(ps.COLUMN, 3.9), layout="constrained")
else:
    fig, ax = ps.figure(height=2.15)
for p, tone, name in ((1.0, ps.GREEN, "$p = 1$"), (0.9, ps.PINK, "$p = 0.9$")):
    ax.plot(CS, curve(p), color=tone.edge, lw=1.8, marker="o", ms=4, mfc=tone.fill, mec=tone.edge, mew=0.9, label=name)
ax.axhline(cap, color=ps.PINK.edge, lw=1.1, ls=(0, (1, 2)), label="$p = 0.9$ as $c \\to \\infty$")
ax.set_xscale("log")
ax.set_xticks([0.2, 1, 10, 100]); ax.set_xticklabels(["0.2", "1", "10", "100"]); ax.minorticks_off()
ax.set_ylim(0, 125); ax.set_yticks([0, 25, 50, 75, 100, 125])
ax.set_xlabel("precision $c = 1/\\tau^2$ (demonstrations)")
ax.set_ylabel("ESS (demonstrations)")
ps.style_axes(ax, grid_axis="y")
ax.legend(loc="upper left")

if sq:  # (b): the same three kinds of description at c = 100, as demonstrations accumulate
    by_n = {}
    for x in csv.DictReader((here / "ess_n_sq.csv").open()):
        by_n.setdefault(x["kind"], []).append((int(x["n"]), float(x["ess"])))
    series = (("reliable", ps.GREEN, "-", "$p = 1$", (100, 117), "bottom"),
              ("unreliable", ps.PINK, "-", "$p = 0.9$", (100, 72), "top"),
              ("partial", ps.GREEN, (0, (4, 2)), "partial description, $p = 1$", (100, 22), "top"))
    for kind, tone, ls, name, at, va in series:
        ns, es = zip(*sorted(by_n[kind]))
        bx.plot(ns, es, color=tone.edge, lw=1.8, ls=ls, marker="o", ms=3.2, mfc=tone.fill if ls == "-" else "white", mec=tone.edge, mew=0.9)
        bx.annotate(name, at, ha="right", va=va, color=tone.edge, fontsize=7.6)
    bx.axvline(16, color=ps.MUTED, lw=0.8, ls=(0, (1, 2)))
    bx.annotate("$n = d$", (16, 62), xytext=(3, 0), textcoords="offset points", ha="left", va="center", color=ps.MUTED, fontsize=7.6)
    bx.set_xlim(-2, 102); bx.set_xticks([0, 25, 50, 75, 100])
    bx.set_ylim(0, 125); bx.set_yticks([0, 25, 50, 75, 100, 125])
    bx.set_xlabel("demonstrations $n$")
    bx.set_ylabel("ESS (demonstrations)")
    ps.style_axes(bx, grid_axis="y")
    ps.panel_labels([ax, bx])
ps.save(fig, str(here / stem))
