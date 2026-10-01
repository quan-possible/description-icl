"""trust.pdf, the paper's figure of the Bayes-optimal predictor's regret
against the trust q it places in a description of true reliability p = 0.9,
at d = 16, a0 = 10, three precisions, alone and with 10 examples in hand.
Flat below p, a cliff at q = 1. Computes trust_curve.csv first and reuses it.

    uv run python results/rq2-reliability/trust_figure.py
"""

import csv
import pathlib

import numpy as np
import orx_figstyle as fs

from descriptor_icl import mixture as mx

here = pathlib.Path(__file__).parent
D, A0, P, K, S = 16, 10.0, 0.9, 40, 8000
QS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999, 1.0)
RS = (0.1, 0.01, 0.001)
NS = (0, 10)

out = here / "trust_curve.csv"
if not out.exists():
    rows = []
    for r in RS:
        paths = mx.simulate(D, A0, r * A0, P, K, S, seed=400)
        for q in QS:
            R = mx.regret(paths, q, 1)
            for n in NS:
                rows.append(dict(d=D, a0=A0, p=P, r=r, q=q, n=n, regret=round(float(R[n]), 4)))
        print(f"r={r} done", flush=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
rows = [{k: float(v) for k, v in x.items()} for x in csv.DictReader(out.open())]

fs.use_style()
fig, ax = fs.figure(width=fs.COLUMN, ratio=0.72)
COLOR = {0.1: fs.PALETTE["orange"], 0.01: fs.PALETTE["blue"], 0.001: fs.PALETTE["purple"]}
STYLE = {0: "-", 10: "--"}
for r in RS:
    for n in NS:
        sel = sorted((x["q"], x["regret"]) for x in rows if x["r"] == r and x["n"] == n)
        X = [max(1 - q, 1e-4) for q, _ in sel]  # q = 1 sits at the far right, labelled 1
        ax.plot(X[:-1], [v for _, v in sel][:-1], STYLE[n], color=COLOR[r], lw=1.4, marker="o", ms=2.5)
        ax.plot(X[-2:], [v for _, v in sel][-2:], ":", color=COLOR[r], lw=1.2, marker="o", ms=2.5)
ax.axvline(1 - P, color=fs.MUTED, lw=0.8, zorder=0)
ax.text(1 - P, 0.09, "$q = p$", color=fs.BASELINE, fontsize=6, ha="center", va="bottom")
ax.set_xscale("log"); ax.invert_xaxis()
ax.set_xticks([0.5, 0.1, 0.01, 0.001, 1e-4]); ax.set_xticklabels(["0.5", "0.9", "0.99", "0.999", "1"]); ax.minorticks_off()
ax.set_xlabel("reliance $q$ on the description")
ax.set_yscale("log"); ax.set_yticks([0.1, 0.3, 1, 3, 10]); ax.set_yticklabels(["0.1", "0.3", "1", "3", "10"]); ax.set_ylim(0.08, 30)
ax.set_ylabel("regret (nats)")
ax.grid(True, axis="y", color=fs.MUTED, lw=0.5)
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=COLOR[r], lw=1.4, label=f"$r = {r:g}$") for r in RS]
handles += [Line2D([], [], color=fs.BASELINE, lw=1.4, ls="-", label="alone"), Line2D([], [], color=fs.BASELINE, lw=1.4, ls="--", label="10 demonstrations in hand")]
ax.legend(handles=handles, loc="upper left", frameon=False, ncol=2, fontsize=6)
fs.save(fig, str(here / "trust"))
