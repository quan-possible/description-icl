"""Write ess_n.csv: each network's ESS with n examples in hand (Definition 1
of the paper) and the Bayes-optimal predictor's on the same prompts, from the
*_regret.csv files; empty where the crossing lies beyond the 20-example grid.

    uv run python results/rq3-meta-trained/ess_n.py
"""

import csv
import pathlib

import numpy as np

from descriptor_icl import gaussian as g

here = pathlib.Path(__file__).parent
rows = []
for stem in sorted(f.name[:-11] for f in here.glob("p*_d5_s*_regret.csv") if "_ptest" not in f.name and "_cont" not in f.name and f.name.split("_")[1] in ("r0.1", "r0.01")):
    reg = list(csv.DictReader((here / f"{stem}_regret.csv").open()))
    for who in ("net", "bayes"):
        plain = np.array([float(x[who]) for x in reg if x["desc"] == "0"])
        desc = np.array([float(x[who]) for x in reg if x["desc"] == "1"])
        for n in range(len(desc)):
            e = float(g.first_crossing(plain, desc[n])[1]) - n
            rows.append(dict(model=stem, who=who, n=n, ess_n="" if np.isnan(e) else round(e, 2)))
with (here / "ess_n.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for stem in ("p0.9_r0.01_d5_s0", "p0.9_r0.1_d5_s0"):
    for who in ("net", "bayes"):
        print(stem, who, {x["n"]: x["ess_n"] for x in rows if x["model"] == stem and x["who"] == who and x["n"] in (0, 5, 10)})
