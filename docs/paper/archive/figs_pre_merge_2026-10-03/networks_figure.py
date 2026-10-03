"""Figure 3 of the paper: meta-trained Transformers against the Bayes-optimal predictor.

(a) ESS with no demonstrations for each trained network, tested at the reliability it was
trained on, beside the Bayes-optimal ESS and the best single Gaussian's (bars; error bars,
standard errors over 20,000 prompts). (b) The networks trained on specific descriptions
(r = 0.01) tested at every reliability: a network's ESS follows its training reliability,
not the test reliability.

Reads results/rq3-meta-trained/summary.csv of the original project (../description-icl);
writes networks.pdf beside this script.

    python3 docs/paper/figs/networks_figure.py
"""

import csv
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42})
here = pathlib.Path(__file__).parent
rows = list(csv.DictReader((here.parents[3] / "description-icl" / "results" / "rq3-meta-trained" / "summary.csv").open()))
P = ["1.0", "0.99", "0.9"]


def row(r, p_train, p_test, seed="0"):
    return next((z for z in rows if z["r"] == r and z["p_train"] == p_train and z["p_test"] == p_test and z["seed"] == seed), None)


fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.4), gridspec_kw=dict(width_ratios=[1.3, 1]))
BAYES, SINGLE, NET = "#8c8c8c", "#c8c8c8", "#1f77b4"
w = 0.27

# (a) each network beside the ESS it should reach and the ESS its output can reach
groups = [(r, p) for r in ("0.01", "0.1") for p in P]
for i, (r, p) in enumerate(groups):
    z = row(r, p, p)
    a.bar(i - w, float(z["calibrated_ess"]), w, color=BAYES, label="Bayes-optimal" if i == 0 else None)
    a.bar(i, float(z["single_ess"] or z["calibrated_ess"]), w, color=SINGLE, label="best single Gaussian" if i == 0 else None)
    a.bar(i + w, float(z["net_ess"]), w, yerr=float(z["net_ess_se"]), color=NET, capsize=2, label="network" if i == 0 else None)
a.set_xticks(range(len(groups)))
a.set_xticklabels([f"$p={p}$" for _, p in groups])
a.axvline(2.5, color="#dddddd", lw=0.8)
a.text(1, -3.6, "specific, $r = 0.01$", ha="center", va="top", clip_on=False)
a.text(4, -3.6, "loose, $r = 0.1$", ha="center", va="top", clip_on=False)
a.set_ylabel("ESS with no demonstrations")
a.set_ylim(0, 17)
a.legend(loc="upper right", frameon=False)

# (b) the specific-description networks tested at every reliability
shades = ["#9ecae1", "#4292c6", "#08519c"]
for j, (p_train, color) in enumerate(zip(P, shades)):
    seed = "1" if p_train == "0.9" else "0"  # the second seed at p = 0.9 was tested at all three reliabilities
    for i, p_test in enumerate(P):
        z = row("0.01", p_train, p_test, seed)
        if z:
            b.bar(i + (j - 1) * w, float(z["net_ess"]), w, yerr=float(z["net_ess_se"]), color=color, capsize=2, label=f"trained at $p={p_train}$" if i == 0 else None)
b.plot(range(len(P)), [float(row("0.01", p, p)["calibrated_ess"]) for p in P], color=BAYES, ls="--", marker="o", ms=3.5, label="Bayes-optimal", zorder=3)
b.set_xticks(range(len(P)))
b.set_xticklabels([f"$p={p}$" for p in P])
b.set_xlabel("reliability of the test prompts")
b.set_ylim(0, 17)
b.legend(loc="upper right", frameon=False)
for ax, letter in ((a, "(a)"), (b, "(b)")):
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.14, 1.02, letter, transform=ax.transAxes, fontweight="bold", va="bottom")
fig.tight_layout(w_pad=1.5, rect=(0, 0.04, 1, 1))
fig.savefig(here / "networks.pdf")
