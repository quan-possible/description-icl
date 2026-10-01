"""Regret against the number of examples: trained network (markers, with
standard errors over 20,000 prompts) and the exact Bayes-optimal learner
(lines), with and without a description, one panel per trained model.
Reads <model>_regret.csv next to this script; writes regret.pdf and
regret.svg.

    uv run python results/rq3-meta-trained/figure.py
"""

import csv
import pathlib

import numpy as np

import orx_figstyle as fs

here = pathlib.Path(__file__).parent
MODELS = [("p1.0_r0.01_d5_s0", "$r = 0.01$, $p = 1$"), ("p1.0_r0.1_d5_s0", "$r = 0.1$, $p = 1$"),
          ("p0.9_r0.01_d5_s0", "$r = 0.01$, $p = 0.9$"), ("p0.9_r0.1_d5_s0", "$r = 0.1$, $p = 0.9$")]
COLOR = {"1": fs.PALETTE["blue"], "0": fs.PALETTE["orange"]}
NAME = {"1": "with description", "0": "without"}

fs.use_style()
fig, axes = fs.figure_grid(2, 2, width=fs.TEXT, ratio=0.75, sharex=True, sharey=True)
for ax, (model, label) in zip(axes.flat, MODELS):
    f = here / f"{model}_regret.csv"
    if not f.exists():
        ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes, color=fs.BASELINE)
        ax.set_title("")
        ax.text(0.97, 0.95, label, ha="right", va="top", transform=ax.transAxes)
        continue
    rows = list(csv.DictReader(f.open()))
    for d in ("1", "0"):
        sub = [x for x in rows if x["desc"] == d]
        n = np.array([int(x["n"]) for x in sub])
        ax.plot(n, [float(x["bayes"]) for x in sub], color=COLOR[d], lw=1.2, zorder=1)
        ax.errorbar(n, [float(x["net"]) for x in sub], yerr=[float(x["net_se"]) for x in sub], fmt="o",
                    ms=3, mfc="white", mec=COLOR[d], ecolor=COLOR[d], color=COLOR[d], lw=0.8, zorder=2)
    ax.text(0.97, 0.95, label, ha="right", va="top", transform=ax.transAxes)
    ax.grid(True, axis="y", color=fs.MUTED, lw=0.5)
fs.panel_labels(axes.flat)
for ax in axes[1]:
    ax.set_xlabel("examples in the prompt, $n$")
    ax.set_xticks([0, 5, 10, 15, 20])
for ax in axes[:, 0]:
    ax.set_ylabel("regret (nats)")
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=COLOR["1"], lw=1.2, label="exact learner, with description"),
           Line2D([], [], color=COLOR["0"], lw=1.2, label="exact learner, without"),
           Line2D([], [], color=fs.BASELINE, marker="o", mfc="white", ms=3, lw=0, label="network")]
fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
fs.save(fig, str(here / "regret"))
