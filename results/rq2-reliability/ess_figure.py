"""ess.pdf, the paper's figure of ESS against precision: (a) reliable
descriptions at d = 4, 16, 64 against (d-1)(1-r); (b) d = 16 at p = 1, 0.99,
0.9 against each reliability's cap, the ESS at which R_ex(n) = (1-p) R_ex(0).
A dense log-spaced grid of r, one seed, a0 = 10; the tables keep the ends.
Computes ess_curve.csv first and reuses it if present.

    uv run python results/rq2-reliability/ess_figure.py
"""

import csv
import pathlib

from matplotlib.lines import Line2D
import numpy as np
import orx_figstyle as fs

from descriptor_icl import gaussian as g
from run import base_G, both_cases, regret_at

here = pathlib.Path(__file__).parent
A0 = 10.0
RS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
PANELS = {4: (1.0,), 16: (1.0, 0.99, 0.9), 64: (1.0,)}  # d -> reliabilities

out = here / "ess_curve.csv"
if not out.exists():
    rows = []
    for d, ps in PANELS.items():
        G0 = base_G(d, A0, 100)
        R_ex = g.regret_examples(G0, 1)
        for p in ps:
            rows.append(dict(d=d, p=p, r=0, ess=round(float(g.first_crossing(R_ex, (1 - p) * R_ex[0])[1]), 3)))  # the cap
        for r in RS:
            cases = both_cases(d, A0, r, 300, 0)
            for p in ps:
                rows.append(dict(d=d, p=p, r=r, ess=round(float(g.first_crossing(R_ex, regret_at(cases, p, p, 1)[0])[1]), 3)))
        print(f"d={d} done", flush=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
rows = [{k: float(v) for k, v in x.items()} for x in csv.DictReader(out.open())]
curve = lambda d, p: [next(x["ess"] for x in rows if x["d"] == d and x["p"] == p and x["r"] == r) for r in RS]
cap = lambda d, p: next(x["ess"] for x in rows if x["d"] == d and x["p"] == p and x["r"] == 0)

fs.use_style()
fig, (a, b) = fs.figure_grid(1, 2, width=fs.TEXT, ratio=0.42)
DCOL = {4: fs.PALETTE["orange"], 16: fs.PALETTE["blue"], 64: fs.PALETTE["purple"]}
for d in (4, 16, 64):
    a.plot(RS, curve(d, 1.0), "-o", ms=3, color=DCOL[d], lw=1.4, label=f"$d = {d}$")
    a.plot(RS, [(d - 1) * (1 - r) for r in RS], "--", color=DCOL[d], lw=1)
PCOL = {1.0: fs.PALETTE["blue"], 0.99: fs.PALETTE["green"], 0.9: fs.PALETTE["red"]}
for p in (1.0, 0.99, 0.9):
    b.plot(RS, curve(16, p), "-o", ms=3, color=PCOL[p], lw=1.4, label=f"$p = {p:g}$")
b.axhline(cap(16, 0.9), color=PCOL[0.9], lw=1, ls=":")
b.text(0.45, cap(16, 0.9) * 1.1, f"cap at $p = 0.9$: {cap(16, 0.9):.0f}", color=PCOL[0.9], fontsize=6, ha="left")
b.text(0.0011, 150, f"cap at $p = 0.99$: {cap(16, 0.99):.0f}", color=PCOL[0.99], fontsize=6, ha="right")
b.set_ylim(3, 200)
for ax in (a, b):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xticks([0.5, 0.1, 0.01, 0.001]); ax.set_xticklabels(["0.5", "0.1", "0.01", "0.001"]); ax.minorticks_off()
    ax.set_xlabel("precision $r$")
    ax.grid(True, axis="y", color=fs.MUTED, lw=0.5)
a.set_ylabel("ESS of the description (examples)")
b.set_ylabel("ESS of the description (examples)")
a.set_yticks([1, 3, 10, 30, 100]); a.set_yticklabels(["1", "3", "10", "30", "100"])
b.set_yticks([3, 10, 30, 100]); b.set_yticklabels(["3", "10", "30", "100"]); b.minorticks_off()
a.text(0.03, 0.95, "reliable, $p = 1$", transform=a.transAxes, va="top")
b.text(0.03, 0.95, "$d = 16$", transform=b.transAxes, va="top")
a.legend(loc="lower right", frameon=False)
b.legend(loc="lower right", frameon=False)
fs.panel_labels([a, b])
handles = [Line2D([], [], color=fs.BASELINE, lw=1.4, marker="o", ms=3, label="exact ESS"),
           Line2D([], [], color=fs.BASELINE, lw=1, ls="--", label="$(d-1)(1-r)$"),
           Line2D([], [], color=fs.BASELINE, lw=1, ls=":", label="cap, $R^{\\mathrm{ex}}(n) = (1-p)R^{\\mathrm{ex}}(0)$")]
fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
fs.save(fig, str(here / "ess"))
