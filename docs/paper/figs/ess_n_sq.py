"""ess_n_sq.csv, the data behind panel (b) of ess_sq.pdf (Figure 2): the Bayes-optimal ESS against the number of
demonstrations n, for three c = 100 descriptions at d = 16, a0 = 10: reliable (p = 1), unreliable (p = 0.9), and a
reliable description of k = 8 of the 16 directions.

The first two reuse ess_curve_sq.regrets on the same seeds, so their n = 0, 1, 10, 100 values equal the rows of
ess_curve_sq.csv (checked below). The partial description's rows are read from partial_sq.csv.

    python3 docs/paper/figs/ess_n_sq.py
"""

import csv
import pathlib

import numpy as np

import ess_curve_sq as E

here = pathlib.Path(__file__).parent
R, K_PARTIAL = 0.001, 8  # c = 1 / (R a0) = 100
NS = (0, 1, 2, 4, 8, 10, 12, 14, 16, 20, 30, 50, 100)

if __name__ == "__main__":
    runs = [E.regrets(R, seed) for seed in range(E.B)]
    R_demo = np.mean([x[0] for x in runs], 0)
    rows = []
    for p, kind in ((1.0, "reliable"), (0.9, "unreliable")):
        R_desc = np.mean([x[1][p] for x in runs], 0)
        for n in NS:
            per = [E.ess(x[0], x[1][p][n], n) for x in runs]
            rows.append(dict(kind=kind, n=n, ess=round(E.ess(R_demo, R_desc[n], n), 3), se=round(float(np.std(per, ddof=1) / np.sqrt(E.B)), 3)))
    old = {(float(x["p"]), int(float(x["n"]))): float(x["worth"]) for x in csv.DictReader((here / "ess_curve_sq.csv").open()) if float(x["r"]) == R}
    for x in rows:  # the same seeds must reproduce the values the paper already quotes
        p = 1.0 if x["kind"] == "reliable" else 0.9
        assert (p, x["n"]) not in old or abs(old[p, x["n"]] - x["ess"]) < 1e-6, (x, old[p, x["n"]])
    for x in csv.DictReader((here / "partial_sq.csv").open()):
        if int(x["k"]) == K_PARTIAL and float(x["r"]) == R and int(x["n"]) in NS:
            rows.append(dict(kind="partial", n=int(x["n"]), ess=float(x["ess"]), se=float(x["se"])))
    with (here / "ess_n_sq.csv").open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    for x in rows:
        print(x)
