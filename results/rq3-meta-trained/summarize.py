"""Collect every evaluated model into summary.csv and print the LaTeX rows
for the paper's network table: ESS of the network and of the exact learner,
and the mean gap in regret (nats) over n = 0..15 with and without a
description.

    uv run python results/rq3-meta-trained/summarize.py
"""

import csv
import pathlib
import re

here = pathlib.Path(__file__).parent
rows = []
for f in sorted(here.glob("p*_d5_s*_ess.csv")):
    e = next(csv.DictReader(f.open()))
    reg = list(csv.DictReader(open(str(f).replace("_ess.csv", "_regret.csv"))))
    gap = {d: sum(float(x["net"]) - float(x["bayes"]) for x in reg if x["desc"] == d) / 16 for d in ("1", "0")}
    rows.append(dict(model=f.stem.replace("_ess", ""), r=e["r"], p_train=e["p_train"], p_test=e["p_test"],
                     net_ess=e["net_ess"], bayes_ess=e["bayes_ess"],
                     gap_desc=round(gap["1"], 3), gap_plain=round(gap["0"], 3)))
with (here / "summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("r & p_train & p_test & ESS net & ESS exact & gap desc & gap plain \\\\")
for x in rows:
    print(f"{x['r']} & {x['p_train']} & {x['p_test']} & {x['net_ess']} & {x['bayes_ess']} & {x['gap_desc']:.3f} & {x['gap_plain']:.3f} \\\\")
