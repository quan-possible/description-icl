"""Worth of a description with n examples already in hand (Definition 2 of
the paper): the further examples an examples-only learner needs to match
the regret of the description plus n examples. Also records the pieces of
Proposition 3: the examples-only regret R_ex(n), the reliable-description
regret R_1(n), and the identification term E[-log pi_n(Z)], the expected
negative log posterior weight on the truth. Writes worth.csv next to this
script. One-step regret, d = 16, a0 = 10, three seeds.

    uv run python results/rq2-reliability/worth.py   # about five minutes
"""

import csv
import pathlib

import numpy as np

from descriptor_icl import gaussian as g
from descriptor_icl import mixture as mx

D, A0, K, S = 16, 10.0, 150, 2000
NS = list(range(0, 41))
rows = []
for seed in (0, 1, 2):
    plain = mx.simulate(D, A0, A0 * 0.999, 1.0, K, S, seed=100 + seed)
    R_ex = mx.regret(plain, 0.0, 1)
    for r in (0.05, 0.001):
        for p in (1.0, 0.9):
            paths = mx.simulate(D, A0, r * A0, p, K, S, seed=200 + seed)
            R_desc = mx.regret(paths, p, 1)
            R_1 = mx.regret(paths, 1.0, 1) if p == 1.0 else None
            ident = -mx.log_weight_true(paths, p).mean(0) if p < 1.0 else np.zeros(K + 1)
            for n in NS:
                rows.append(dict(seed=seed, d=D, a0=A0, r=r, p=p, n=n, R_ex=round(R_ex[n], 4),
                                 R_desc=round(R_desc[n], 4), ident=round(float(ident[n]), 4),
                                 worth=round(float(g.first_crossing(R_ex, R_desc[n])[1]) - n, 2)))
with (pathlib.Path(__file__).parent / "worth.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in (0.05, 0.001):
    for p in (1.0, 0.9):
        sel = [x for x in rows if x["r"] == r and x["p"] == p and x["n"] in (0, 1, 3, 8, 16, 24, 40)]
        by_n = {}
        for x in sel:
            by_n.setdefault(x["n"], []).append(x["worth"])
        print(f"r={r:<6} p={p:<4}", {n: round(float(np.mean(v)), 1) for n, v in by_n.items()})
