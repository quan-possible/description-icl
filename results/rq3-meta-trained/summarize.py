"""Collect every evaluated model into summary.csv and write the body of the
paper's network table, docs/paper/figs/networks_table.tex: the ESS of the network, of the exact
learner with the network's training reliability (the predictor the network
was trained toward), and of the calibrated exact learner (targets.csv), plus
the mean gap in squared error over n = 0..20 with and without a description.

    uv run python results/rq3-meta-trained/summarize.py
"""

import csv
import pathlib

here = pathlib.Path(__file__).parent
targets = {(x["r"], x["p"]): x["ess"] for x in csv.DictReader((here / "targets.csv").open()) if x["ess"]}
trust_q = {x["model"]: x["q"] for x in csv.DictReader((here / "trust_q.csv").open())} if (here / "trust_q.csv").exists() else {}
def _key(f):  # training reliability descending, precision descending, test reliability descending, then seed
    stem = f.name[:-8]; head, _, ptest = stem.partition("_ptest")
    p, r, _, s = head.split("_"); return (-float(p[1:]), -float(r[1:]), -(float(ptest) if ptest else float(p[1:])), int(s[1:]))
FILES = [f for f in here.glob("p*_d5_s*_ess.csv") if f.name.split("_")[1] in ("r0.1", "r0.01") and "_cont" not in f.name]
ORDER = [f.name[:-8] for f in sorted(FILES, key=_key)]
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
    # implied trust with no demonstrations: the closed form of trust_q.py where computed, else evaluate.py's grid
    reliance = trust_q.get(stem.split("_ptest")[0]) or next(x["implied_q"] for x in csv.DictReader((here / f"{stem}_trust.csv").open()) if x["n"] == "0")
    rows.append(dict(model=stem, seed=stem.split("_")[3][1:2], r=e["r"], p_train=e["p_train"], p_test=e["p_test"],
                     net_ess=float(e["net_ess"]), net_ess_se=round(se0 / abs(plain[k] - plain[k + 1]), 2),
                     trained_ess=float(targets[(e["r"], e["p_test"])]) if diag else float(e["bayes_ess"]),
                     reliance=float(reliance),
                     calibrated_ess=float(targets[(e["r"], e["p_test"])]),
                     bayes_ess_prompts=float(e["bayes_ess"]),  # the trained Bayes-optimal ESS on the network's own prompts (paired)
                     gap_desc=round(gap["1"], 3), gap_plain=round(gap["0"], 3)))
with (here / "summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
table = here.parents[1] / "docs" / "paper" / "figs" / "networks_table.tex"  # the paper's Table 2: one row per setting, over seeds
groups = {}
for x in rows:
    groups.setdefault((x["p_train"], x["r"], x["p_test"]), []).append(x)
with table.open("w") as f:
    f.write("\\begin{tabular}{rrrcrrrr}\n\\toprule\n"
            "& & & & \\multicolumn{3}{c}{ESS} & \\\\\n"
            "\\cmidrule(lr){5-7}\n"
            "$r$ & $p_{\\mathrm{train}}$ & $p_{\\mathrm{test}}$ & Seeds & Network & Trained & Calibrated & Trust \\\\\n")
    last = None
    for (pt, r, ps_), xs in groups.items():
        if pt != last:  # a rule above each training reliability
            f.write("\\midrule\n"); last = pt
        x, ess = xs[0], [z["net_ess"] for z in xs]
        cal = f"{x['calibrated_ess']:.2f}" if pt != ps_ else "---"
        trained = f"{x['trained_ess']:.2f}" if x["trained_ess"] > 0 else "$\\le 0$"
        # one seed: the standard error over prompts; several: the standard deviation across seeds
        spread = x["net_ess_se"] if len(xs) == 1 else float(sum((e - sum(ess) / len(ess)) ** 2 for e in ess) / (len(ess) - 1)) ** 0.5
        mean = sum(ess) / len(ess)
        net = f"{mean:.2f} $\\pm$ {spread:.2f}" if mean > 0 else "$\\le 0$"
        trust = sum(z["reliance"] for z in xs) / len(xs)
        f.write(f"{r} & {float(pt):g} & {float(ps_):g} & {len(xs)} & {net} & {trained} & {cal} & {trust:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print(table.read_text())
