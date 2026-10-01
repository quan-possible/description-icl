"""Write precision_table.tex, the body of the paper's Table 1, from ess.csv:
the one-step ESS of a reliable description at a0 = 10 against the additive
rule (d-1)(1-r) + 1/(r a0) of Proposition 1.

    uv run python results/rq1-single-query-gap/table.py
"""

import csv
import pathlib

here = pathlib.Path(__file__).parent
rows = [x for x in csv.DictReader((here / "ess.csv").open()) if float(x["a0"]) == 10.0 and int(x["d"]) in (4, 16, 64)]
with (here / "precision_table.tex").open("w") as f:
    f.write("\\begin{tabular}{rrrr}\n\\toprule\n$d$ & $r$ & $\\ess$ & $(d-1)(1-r) + 1/(r a_0)$ \\\\\n\\midrule\n")
    for x in sorted(rows, key=lambda x: (int(x["d"]), -float(x["r"]))):
        d, r, e = int(x["d"]), float(x["r"]), float(x["ess_N1"])
        f.write(f"{d} & {r:g} & {e:.3g} & {(d - 1) * (1 - r) + 1 / (r * 10.0):.3g} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print((here / "precision_table.tex").read_text())
