"""networks_ess_n.pdf, the appendix figure: each meta-trained network's ESS with n demonstrations in hand, against the
Bayes-optimal predictor's on the same prompts, trained and tested at the same reliability p; rows, the specific
(r = 0.01) and the loose (r = 0.1) description; one line per training seed. A curve stops where the crossing lies
beyond the 20-demonstration grid; the dotted line, 20 - n, is the largest ESS the grid can show.

Reads results/rq3-meta-trained/ess_n.csv (ess_n.py); writes networks_ess_n.pdf beside this script.

    python3 docs/paper/figs/networks_ess_n.py
"""

import csv
import pathlib

import numpy as np

import paper_style as ps

here = pathlib.Path(__file__).parent
rows = list(csv.DictReader((here.parents[2] / "results" / "rq3-meta-trained" / "ess_n.csv").open()))
P, R = ["1.0", "0.99", "0.9"], [("0.01", "specific"), ("0.1", "loose")]
N = np.arange(21)


def curve(model, who):
    """ESS_n at n = 0..20, nan where the crossing lies beyond the grid."""
    v = {int(z["n"]): z["ess_n"] for z in rows if z["model"] == model and z["who"] == who}
    return np.array([float(v[n]) if v.get(n) else np.nan for n in N])


ps.use_style()
fig, axes = ps.plt.subplots(2, 3, figsize=(ps.WIDE, 3.3), layout="constrained", sharex=True, sharey="row")
for i, (r, name) in enumerate(R):
    for j, p in enumerate(P):
        ax = axes[i, j]
        seeds = sorted({z["model"] for z in rows if z["model"].startswith(f"p{p}_r{r}_d5_s")})
        ax.plot(N, 20 - N, color=ps.SUBTLE, lw=0.9, ls=":")
        ax.plot(N, curve(seeds[0], "bayes"), zorder=4, **ps.BAYES)  # the seeds share their evaluation prompts
        for s in seeds:
            ax.plot(N, curve(s, "net"), color=ps.BLUE.edge, lw=1.1, marker="o", ms=2.2, mfc=ps.BLUE.fill, mec=ps.BLUE.edge, mew=0.7)
        ax.set_title(f"{name}, $p = {float(p):g}$", fontsize=8, color=ps.TEXT, loc="left")
        ax.set_xlim(0, 20); ax.set_xticks([0, 5, 10, 15, 20])
        ps.style_axes(ax, grid_axis="y")
    axes[i, 0].set_ylabel("ESS$_n$ (demonstrations)")
    axes[i, 0].set_ylim(0, 20 if r == "0.01" else 7)
for ax in axes[1]:
    ax.set_xlabel("demonstrations in hand, $n$")
from matplotlib.lines import Line2D

handles = [Line2D([], [], color=ps.BLUE.edge, lw=1.1, marker="o", ms=2.2, mfc=ps.BLUE.fill, label="network (one line per seed)"),
           Line2D([], [], label="Bayes-optimal", **ps.BAYES), Line2D([], [], color=ps.SUBTLE, lw=0.9, ls=":", label="grid limit, $20 - n$")]
fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
ps.save(fig, str(here / "networks_ess_n"))
