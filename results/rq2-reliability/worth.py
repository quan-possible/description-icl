"""Worth of a description with n examples already in hand (Definition 2 of
the paper): the further examples an examples-only learner needs to match
the regret of the description plus n examples. Also records the pieces of
Proposition 3: the examples-only regret R_ex(n), the reliable-description
regret R_1(n), and the identification term E[-log pi_n(Z)], the expected
negative log posterior weight on the truth. Rows with q = 1 at p = 0.9 are the fully
trusting learner, which takes the description as certainly correct. Writes
worth.csv next to this script. One-step regret, d = 16, a0 = 10, three seeds.

    uv run python results/rq2-reliability/worth.py   # about five minutes
"""

import csv
import pathlib

import numpy as np

from description_icl import gaussian as g
from description_icl import mixture as mx

D, A0, K, S = 16, 10.0, 260, 8000
NS = list(range(0, 101))
rows = []
for seed in (0, 1, 2):
    plain = mx.simulate(D, A0, A0 * 0.999, 1.0, K, S, seed=100 + seed)
    R_ex = mx.regret(plain, 0.0, 1)
    for r in (0.1, 0.01, 0.001):
        for p in (1.0, 0.9):
            paths = mx.simulate(D, A0, r * A0, p, K, S, seed=200 + seed)
            ident = -mx.log_weight_true(paths, p).mean(0) if p < 1.0 else np.zeros(K + 1)
            for q in ((p, 1.0) if p < 1.0 else (p,)):
                R_desc = mx.regret(paths, q, 1)
                for n in NS:
                    hit = R_desc[n] <= R_ex[0]  # otherwise worse than no information at all
                    rows.append(dict(seed=seed, d=D, a0=A0, r=r, p=p, q=q, n=n, R_ex=round(R_ex[n], 4),
                                     R_desc=round(R_desc[n], 4), ident=round(float(ident[n]), 4) if q == p else "",
                                     worth=round(float(g.first_crossing(R_ex, R_desc[n])[1]) - n, 2) if hit else ""))
with (pathlib.Path(__file__).parent / "worth.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in (0.1, 0.01, 0.001):
    for p, q in ((1.0, 1.0), (0.9, 0.9), (0.9, 1.0)):
        sel = [x for x in rows if x["r"] == r and x["p"] == p and x["q"] == q and x["n"] in (0, 1, 10, 100)]
        by_n = {}
        for x in sel:
            by_n.setdefault(x["n"], []).append(x["worth"] if x["worth"] != "" else np.nan)
        print(f"r={r:<6} p={p:<4} q={q:<4}", {n: round(float(np.mean(v)), 1) for n, v in by_n.items()},
              "regret", {n: round(float(np.mean([x["R_desc"] for x in sel if x["n"] == n])), 2) for n in (0, 1, 10, 100)})
