"""Worth of a description against the examples already in hand. (a) The
exact learner at d = 16, a0 = 10 (worth.csv, mean of three seeds): a
reliable description's worth falls or stays flat as examples accumulate; an
unreliable precise one gains worth once examples can check it. (b) The
trained networks at d = 5 (results/rq3-meta-trained/*_regret.csv) against
the exact learner, r = 0.05. Writes worth.pdf and worth.svg.

    uv run python results/rq2-reliability/worth_figure.py
"""

import csv
import pathlib

import numpy as np
from matplotlib.lines import Line2D

import orx_figstyle as fs
from descriptor_icl import gaussian as g

here = pathlib.Path(__file__).parent
rq3 = here.parent / "rq3-meta-trained"
COLOR = {0.05: fs.PALETTE["blue"], 0.001: fs.PALETTE["purple"], 0.5: fs.PALETTE["orange"]}
STYLE = {1.0: "-", 0.9: "--"}

fs.use_style()
fig, (a, b) = fs.figure_grid(1, 2, width=fs.TEXT, ratio=0.42)

rows = [{k: float(v) for k, v in x.items()} for x in csv.DictReader((here / "worth.csv").open())]
for r in (0.001, 0.05):
    for p in (1.0, 0.9):
        sel = [x for x in rows if x["r"] == r and x["p"] == p]
        ns = sorted({x["n"] for x in sel})
        w = [np.nanmean([x["worth"] for x in sel if x["n"] == n]) for n in ns]
        a.plot(ns, w, STYLE[p], color=COLOR[r], lw=1.4)
a.set_xlabel("examples in hand, $n$")
a.set_ylabel("worth of the description (examples)")
a.set_yscale("log")
a.set_yticks([3, 10, 30, 100]); a.set_yticklabels(["3", "10", "30", "100"])
a.text(0.03, 0.95, "$d = 16$, exact learner", transform=a.transAxes, va="top")
a.grid(True, axis="y", color=fs.MUTED, lw=0.5)

def worth_curve(stem, who):
    reg = list(csv.DictReader((rq3 / f"{stem}_regret.csv").open()))
    plain = np.array([float(x[who]) for x in reg if x["desc"] == "0"])
    desc = np.array([float(x[who]) for x in reg if x["desc"] == "1"])
    return [float(g.first_crossing(plain, desc[n])[1]) - n for n in range(12)]

for stem, p in (("p1.0_r0.05_d5_s0", 1.0), ("p0.9_r0.05_d5_s0", 0.9)):
    b.plot(range(12), worth_curve(stem, "bayes"), STYLE[p], color=COLOR[0.05], lw=1.4, zorder=1)
    b.plot(range(12), worth_curve(stem, "net"), "o", ms=3, mfc="white", mec=COLOR[0.05], zorder=2)
b.set_xlabel("examples in hand, $n$")
b.set_ylabel("worth of the description (examples)")
b.text(0.03, 0.95, "$d = 5$, $r = 0.05$, networks", transform=b.transAxes, va="top")
b.grid(True, axis="y", color=fs.MUTED, lw=0.5)
fs.panel_labels([a, b])
handles = [Line2D([], [], color=COLOR[0.001], lw=1.4, label="$r = 0.001$"),
           Line2D([], [], color=COLOR[0.05], lw=1.4, label="$r = 0.05$"),
           Line2D([], [], color=fs.BASELINE, lw=1.4, ls="-", label="reliable, $p = 1$"),
           Line2D([], [], color=fs.BASELINE, lw=1.4, ls="--", label="unreliable, $p = 0.9$"),
           Line2D([], [], color=fs.BASELINE, marker="o", mfc="white", ms=3, lw=0, label="network")]
fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
fs.save(fig, str(here / "worth"))
