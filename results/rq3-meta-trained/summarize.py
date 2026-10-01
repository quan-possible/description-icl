"""Collect every evaluated model into summary.csv and write networks_table.tex,
the body of the paper's network table: the ESS of the network, of the exact
learner with the network's training reliability (the predictor the network
was trained toward), and of the calibrated exact learner (targets.csv), plus
the mean gap in regret (nats) over n = 0..20 with and without a description.

    uv run python results/rq3-meta-trained/summarize.py
"""

import csv
import pathlib

here = pathlib.Path(__file__).parent
targets = {(x["r"], x["p"]): x["ess"] for x in csv.DictReader((here / "targets.csv").open()) if x["ess"]}
def _key(f):  # training reliability descending, precision descending, seed, then test reliability descending
    stem = f.name[:-8]; head, _, ptest = stem.partition("_ptest")
    p, r, _, s = head.split("_"); return (-float(p[1:]), -float(r[1:]), int(s[1:]), -(float(ptest) if ptest else float(p[1:])))
FILES = [f for f in here.glob("p*_d5_s*_ess.csv") if f.name.split("_")[1] in ("r0.1", "r0.01") and "_cont" not in f.name]
ORDER = [f.name[:-8] for f in sorted(FILES, key=_key)]
single = {(x["r"], x["p"]): x["ess_n"] for x in csv.DictReader((here / "single_gaussian.csv").open()) if x["who"] == "single" and x["n"] == "0"}
rows = []
for stem in ORDER:
    f = here / f"{stem}_ess.csv"
    if not f.exists():
        continue
    e = next(csv.DictReader(f.open()))
    reg = list(csv.DictReader((here / f"{stem}_regret.csv").open()))
    K = len(reg) // 2
    gap = {d: sum(float(x["net"]) - float(x["bayes"]) for x in reg if x["desc"] == d) / K for d in ("1", "0")}
    # network ESS standard error: both regrets' standard errors over the local slope of its own curve
    plain = [float(x["net"]) for x in reg if x["desc"] == "0"]
    plain_se = [float(x["net_se"]) for x in reg if x["desc"] == "0"]
    se0 = float(next(x["net_se"] for x in reg if x["desc"] == "1" and x["n"] == "0"))
    k = min(max(int(float(e["net_ess"])), 0), K - 2)
    se0 = (se0 ** 2 + plain_se[k] ** 2) ** 0.5
    diag = e["p_train"] == e["p_test"]  # on its own distribution the trained predictor is the calibrated one: one number, from targets.csv
    reliance = next(x["implied_q"] for x in csv.DictReader((here / f"{stem}_trust.csv").open()) if x["n"] == "0")
    rows.append(dict(model=stem, seed=stem.split("_")[3][1:2], r=e["r"], p_train=e["p_train"], p_test=e["p_test"],
                     net_ess=float(e["net_ess"]), net_ess_se=round(se0 / abs(plain[k] - plain[k + 1]), 2),
                     trained_ess=float(targets[(e["r"], e["p_test"])]) if diag else float(e["bayes_ess"]),
                     reliance=float(reliance),
                     calibrated_ess=float(targets[(e["r"], e["p_test"])]),
                     single_ess=float(single[(e["r"], e["p_test"])]) if diag and (e["r"], e["p_test"]) in single else "",
                     gap_desc=round(gap["1"], 3), gap_plain=round(gap["0"], 3)))
with (here / "summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with (here / "networks_table.tex").open("w") as f:
    f.write("\\begin{tabular}{rrrrrrrrrr}\n\\toprule\n"
            "& & & \\multicolumn{4}{c}{ESS} & & \\multicolumn{2}{c}{Regret gap (nats)} \\\\\n"
            "\\cmidrule(lr){4-7}\\cmidrule(lr){9-10}\n"
            "$r$ & $p_{\\mathrm{train}}$ & $p_{\\mathrm{test}}$ & Network & Trained & Calibrated & Single Gaussian & Reliance & Description & None \\\\\n"
            "\\midrule\n")
    for x in rows:
        cal = f"{x['calibrated_ess']:.2f}" if x["p_train"] != x["p_test"] else "---"
        trained = f"{x['trained_ess']:.2f}" if x["trained_ess"] > 0 else "$\\le 0$"
        net = f"{x['net_ess']:.2f} $\\pm$ {x['net_ess_se']:.2f}" if x["net_ess"] > 0 else "$\\le 0$"
        sg = f"{x['single_ess']:.2f}" if x["single_ess"] != "" else "---"
        f.write(f"{x['r']} & {float(x['p_train']):g} & {float(x['p_test']):g} & {net} & "
                f"{trained} & {cal} & {sg} & {x['reliance']:.3g} & {abs(x['gap_desc']) if abs(x['gap_desc']) < 5e-4 else x['gap_desc']:.3f} & {x['gap_plain']:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print(open(here / "networks_table.tex").read())
