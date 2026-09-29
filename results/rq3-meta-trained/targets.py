"""The exact learner's regret and ESS at the RQ3 settings, which the trained
networks should reproduce. Writes targets.csv next to this script.

    uv run python results/rq3-meta-trained/targets.py   # about a minute
"""

import csv
import pathlib

from descriptor_icl import gaussian as g
from descriptor_icl import mixture as mx
from evaluate import bayes_paths, draw, regret

D, A0, K, S = 5, 10.0, 16, 200_000

rows = []
for r in (0.5, 0.05):
    plain = draw(S, K, D, A0, r, 1.0, 0.0, seed=1)
    curve = regret(bayes_paths(plain, A0, r).mu0, plain)[0]
    for p in (1.0, 0.9):
        batch = draw(S, K, D, A0, r, p, 1.0, seed=10)
        reg, se = regret(mx.mean_prediction(bayes_paths(batch, A0, r), p), batch)
        rows.append(dict(d=D, a0=A0, r=r, p=p, regret=round(reg[0], 3), regret_se=round(se[0], 3),
                         ess=round(g.first_crossing(curve, reg[0])[1], 2)))
rows.append(dict(d=D, a0=A0, r="", p="", regret="", regret_se="", ess=""))
rows += [dict(d=D, a0=A0, r="none", p=f"n={n}", regret=round(curve[n], 3), regret_se="", ess="")
         for n in range(K)]
with (pathlib.Path(__file__).parent / "targets.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(*rows, sep="\n")
