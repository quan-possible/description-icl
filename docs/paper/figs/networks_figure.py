"""networks.pdf, Figure 3 of the paper: meta-trained Transformers against the Bayes-optimal predictor, for the
specific description (r = 0.01), d = 5, a0 = 10, ESS with no demonstrations (the loose description is in Table 2).

(a) Trained and tested at the same reliability p: paired bars for the Bayes-optimal ESS on the network's prompts and the network's
(the network's ESS is its mean over training seeds; error bars, the standard deviation across seeds); the ESS is in
squared error.
(b) Trained at one reliability, tested at another: each cell is the network's ESS as a share of the
Bayes-optimal ESS for the test prompts, from the seed mean. The diagonal is train = test.

Style: paper_style.py (the Editorial Neutral binding). Reads results/rq3-meta-trained/summary.csv at the
repository root; writes networks.pdf beside this script.

    python3 docs/paper/figs/networks_figure.py
"""

import csv
import pathlib

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import paper_style as ps

here = pathlib.Path(__file__).parent
rows = list(csv.DictReader((here.parents[2] / "results" / "rq3-meta-trained" / "summary.csv").open()))
P = ["1.0", "0.99", "0.9"]


def row(p_train, p_test):
    """The setting over its seeds: mean network ESS, and its spread (standard deviation across seeds, or the
    standard error over prompts when there is one seed)."""
    zs = [z for z in rows if z["r"] == "0.01" and z["p_train"] == p_train and z["p_test"] == p_test]
    ess = np.array([float(z["net_ess"]) for z in zs])
    spread = float(zs[0]["net_ess_se"]) if len(zs) == 1 else float(ess.std(ddof=1))
    return dict(zs[0], net_ess=float(ess.mean()), net_ess_se=spread, seeds=len(zs))


ps.use_style()
fig, (a, b) = ps.figure(2, width=ps.WIDE, height=2.2, gridspec_kw=dict(width_ratios=[1.25, 1]))

# (a) trained and tested at the same reliability
w = 0.36
for i, p in enumerate(P):
    z = row(p, p)
    bay, net, se = float(z["bayes_ess_prompts"]), float(z["net_ess"]), float(z["net_ess_se"])  # paired: same prompts
    a.bar(i - w / 2, bay, w, color=ps.GRAY.fill, edgecolor=ps.GRAY.edge, lw=0.9, label="Bayes-optimal" if i == 0 else None)
    a.bar(i + w / 2, net, w, color=ps.BLUE.fill, edgecolor=ps.BLUE.edge, lw=0.9, label="network" if i == 0 else None)
    a.errorbar(i + w / 2, net, yerr=se, color=ps.BLUE.edge, lw=1)
a.set_xticks(range(3)); a.set_xticklabels([f"{float(p):g}" for p in P])
a.set_xlabel("$p$, in training and test")
a.set_ylabel("ESS (demonstrations)")
a.set_ylim(0, 19); a.set_yticks([0, 5, 10, 15])
a.legend(loc="upper right", ncol=1)
ps.style_axes(a, grid_axis="y")

# (b) share of the calibrated ESS, trained (rows) against tested (columns)
TEST = ["1.0", "0.9"]
share = np.full((3, 2), np.nan)
for i, pt in enumerate(P):
    for j, ps_ in enumerate(TEST):
        z = row(pt, ps_)
        share[i, j] = max(float(z["net_ess"]), 0) / float(z["calibrated_ess"])
cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff", ps.BLUE.edge])
b.imshow(share, cmap=cmap, vmin=0, vmax=1.15, aspect="auto")
for i in range(3):
    for j in range(2):
        txt = "≤0%" if float(row(P[i], TEST[j])["net_ess"]) <= 0 else f"{share[i, j]:.0%}"
        b.text(j, i, txt, ha="center", va="center", color=ps.TEXT, fontsize=8.4)
for k in range(2):  # outline the cells where training matches the test
    i = P.index(TEST[k])
    b.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((k - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ps.TEXT, lw=1.2))
b.set_xticks(range(2)); b.set_xticklabels([f"{float(p):g}" for p in TEST])
b.set_yticks(range(3)); b.set_yticklabels([f"{float(p):g}" for p in P])
b.set_xlabel("$p$ in test"); b.set_ylabel("$p$ in training")
b.tick_params(length=0, pad=6)
for side in b.spines.values():
    side.set_visible(False)
ps.panel_labels((a, b), titles=["", "share of the calibrated ESS"])
ps.save(fig, str(here / "networks"))
