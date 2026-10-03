"""networks.pdf, Figure 3 of the paper: meta-trained Transformers against the Bayes-optimal predictor,
ESS with no demonstrations, d = 5, a0 = 10; error bars, standard errors over 20,000 prompts.

(a) Each network tested at the reliability it was trained on (blue bars); dashed, the Bayes-optimal ESS;
dotted, the best single Gaussian where the Bayes-optimal predictive is a mixture of two Gaussians.
(b) The networks trained on specific descriptions (r = 0.01), each tested on prompts whose descriptions
are all relevant (green, test p = 1) and on prompts where one in ten is irrelevant (pink, test p = 0.9);
dashed, the Bayes-optimal ESS for those test prompts.

Style: paper_style.py (the Editorial Neutral binding). Reads results/rq3-meta-trained/summary.csv of the
original project (../description-icl); writes networks.pdf beside this script.

    python3 docs/paper/figs/networks_figure.py
"""

import csv
import pathlib

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import paper_style as ps

here = pathlib.Path(__file__).parent
rows = list(csv.DictReader((here.parents[3] / "description-icl" / "results" / "rq3-meta-trained" / "summary.csv").open()))
P = ["1.0", "0.99", "0.9"]


def row(r, p_train, p_test):
    return next(z for z in rows if z["r"] == r and z["p_train"] == p_train and z["p_test"] == p_test and z["seed"] == "0")


def bar(ax, x, z, tone, w):
    v = float(z["net_ess"])
    b = ax.bar(x, max(v, 0), w, color=tone.fill, edgecolor=tone.edge, lw=0.9, zorder=2)
    if v > 0:
        ax.errorbar(x, v, yerr=float(z["net_ess_se"]), color=tone.edge, lw=1, zorder=3)
    ax.hlines(float(z["calibrated_ess"]), x - w / 2 - 0.06, x + w / 2 + 0.06, zorder=4, **ps.BAYES)
    return b


ps.use_style()
fig, (a, b) = ps.figure(2, width=ps.WIDE, height=2.15, sharey=True, gridspec_kw=dict(width_ratios=[1.3, 1]))

# (a) each network at its own training reliability
xa = [0, 1.2, 2.4, 4.0, 5.2, 6.4]
bars = []
for x, (r, p) in zip(xa, [(r, p) for r in ("0.1", "0.01") for p in P]):
    z = row(r, p, p)
    bars.append(bar(a, x, z, ps.BLUE, 0.62))
    if z["single_ess"]:
        a.hlines(float(z["single_ess"]), x - 0.37, x + 0.37, color=ps.SUBTLE, lw=1.3, ls=(0, (1, 1.5)), zorder=4)
a.set_xticks(xa); a.set_xticklabels([f"{float(p):g}" for p in P] * 2)
a.set_xlabel("$p$, in training and test", labelpad=15)
sec = a.secondary_xaxis(-0.13)
sec.set_xticks([1.2, 5.2]); sec.set_xticklabels(["loose, $r = 0.1$", "specific, $r = 0.01$"])
sec.tick_params(length=0, colors=ps.MUTED); sec.spines["bottom"].set_visible(False)
a.set_ylabel("ESS (demonstrations)")
a.legend(handles=[Patch(facecolor=ps.BLUE.fill, edgecolor=ps.BLUE.edge, label="network"), Line2D([], [], label="Bayes-optimal", **ps.BAYES),
                  Line2D([], [], color=ps.SUBTLE, lw=1.3, ls=(0, (1, 1.5)), label="best single Gaussian")], loc="upper left", ncol=3, handlelength=1.6, columnspacing=1.2)

# (b) the specific-description networks on all-relevant and one-in-ten-irrelevant prompts
w = 0.36
for i, p in enumerate(P):
    for k, (pt, tone) in enumerate((("1.0", ps.GREEN), ("0.9", ps.PINK))):
        bars.append(bar(b, i + (k - 0.5) * w * 1.1, row("0.01", p, pt), tone, w))
b.set_xticks(range(3)); b.set_xticklabels([f"{float(p):g}" for p in P])
b.set_xlabel("$p$ in training", labelpad=15)
b.legend(handles=[Patch(facecolor=ps.GREEN.fill, edgecolor=ps.GREEN.edge, label="test $p = 1$"), Patch(facecolor=ps.PINK.fill, edgecolor=ps.PINK.edge, label="test $p = 0.9$"),
                  Line2D([], [], label="Bayes-optimal", **ps.BAYES)], loc="upper right", ncol=3, handlelength=1.6, columnspacing=1.2)
for ax in (a, b):
    ax.set_ylim(0, 20); ax.set_yticks([0, 5, 10, 15])
    ps.style_axes(ax, grid_axis="y")
fig.canvas.draw()
for ax, bc in ((a, bars[:6]), (b, bars[6:])):
    for c in bc:
        ps.soften_bar(c, ax, fill=c.patches[0].get_facecolor(), edge=c.patches[0].get_edgecolor(), rounding_size=3)
ps.panel_labels((a, b))
ps.save(fig, str(here / "networks"))
