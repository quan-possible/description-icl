"""Collect every evaluated model into summary.csv and write networks_table.tex,
the body of the paper's network table: the ESS of the network and of the
exact learner calibrated to the test reliability (targets.csv), and the mean
gap in regret (nats) over n with and without a description.

    uv run python results/rq3-meta-trained/summarize.py
"""

import csv
import pathlib

here = pathlib.Path(__file__).parent
targets = {(x["r"], x["p"]): x["ess"] for x in csv.DictReader((here / "targets.csv").open()) if x["ess"]}
ORDER = ["p1.0_r0.1_d5_s0", "p1.0_r0.01_d5_s0", "p0.9_r0.1_d5_s0", "p0.9_r0.01_d5_s0",
         "p1.0_r0.1_d5_s0_ptest0.9", "p1.0_r0.01_d5_s0_ptest0.9"]
rows = []
for stem in ORDER:
    f = here / f"{stem}_ess.csv"
    if not f.exists():
        continue
    e = next(csv.DictReader(f.open()))
    reg = list(csv.DictReader((here / f"{stem}_regret.csv").open()))
    K = len(reg) // 2
    gap = {d: sum(float(x["net"]) - float(x["bayes"]) for x in reg if x["desc"] == d) / K for d in ("1", "0")}
    rows.append(dict(model=stem, r=e["r"], p_train=e["p_train"], p_test=e["p_test"],
                     net_ess=float(e["net_ess"]), exact_ess=float(targets[(e["r"], e["p_test"])]),
                     trained_trust_ess=float(e["bayes_ess"]),
                     gap_desc=round(gap["1"], 3), gap_plain=round(gap["0"], 3)))
with (here / "summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with (here / "networks_table.tex").open("w") as f:
    f.write("\\begin{tabular}{rrrrrrr}\n\\toprule\n"
            "& & & \\multicolumn{2}{c}{ESS} & \\multicolumn{2}{c}{Regret gap (nats)} \\\\\n"
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "$r$ & $p_{\\mathrm{train}}$ & $p_{\\mathrm{test}}$ & Network & Exact & Descriptor & None \\\\\n"
            "\\midrule\n")
    for x in rows:
        gd = abs(x["gap_desc"]) if abs(x["gap_desc"]) < 5e-4 else x["gap_desc"]
        f.write(f"{x['r']} & {float(x['p_train']):g} & {float(x['p_test']):g} & {x['net_ess']:.1f} & "
                f"{x['exact_ess']:.1f} & {gd:.3f} & {x['gap_plain']:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print(open(here / "networks_table.tex").read())
