"""Write trust_curve.csv: the Bayes-optimal predictor's one-step regret at d = 5,
a0 = 10 as a function of the trust q it places in a description of true
reliability p, with n examples in hand. Read next to each network's implied
trust: the curve is asymmetric around q = p, cheap below it and steep above.

    uv run python results/rq3-meta-trained/trust_curve.py
"""

import csv
import pathlib

from description_icl import mixture as mx

here = pathlib.Path(__file__).parent
D, A0, K, S = 5, 10.0, 24, 20_000
QS = (0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999, 1.0)
rows = []
for p in (0.9, 0.99, 1.0):
    for r in (0.1, 0.01):
        paths = mx.simulate(D, A0, r * A0, p, K, S, seed=300)
        for q in QS:
            R = mx.regret(paths, q, 1)
            for n in (0, 2, 5, 10):
                rows.append(dict(d=D, a0=A0, p=p, r=r, q=q, n=n, regret=round(float(R[n]), 4)))
        print(f"p={p} r={r}", {q: round(float(mx.regret(paths, q, 1)[0]), 3) for q in QS}, flush=True)
with (here / "trust_curve.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
