"""Write reliability_table.tex, the body of the paper's Table 2, from
ess_map.csv: the one-step ESS at d = 16, a0 = 10 over precision and
reliability.

    uv run python results/rq2-reliability/table.py
"""

import csv
import pathlib

here = pathlib.Path(__file__).parent
rows = [x for x in csv.DictReader((here / "ess_map.csv").open())
        if int(x["d"]) == 16 and float(x["a0"]) == 10.0 and int(x["N"]) == 1]
ps = sorted({float(x["p"]) for x in rows}, reverse=True)
rs = sorted({float(x["r"]) for x in rows}, reverse=True)
with (here / "reliability_table.tex").open("w") as f:
    f.write("\\begin{tabular}{r" + "r" * len(ps) + "}\n\\toprule\n$r$ & " + " & ".join(f"$p = {p:g}$" for p in ps) + " \\\\\n\\midrule\n")
    for r in rs:
        vals = [next(float(x["ess"]) for x in rows if float(x["r"]) == r and float(x["p"]) == p) for p in ps]
        f.write(f"{r:g} & " + " & ".join(f"{v:.3g}" for v in vals) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print((here / "reliability_table.tex").read_text())
