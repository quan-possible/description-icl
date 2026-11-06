"""objective_main_curve.py: a second reading of the objective ablation's ESS (see objective_analysis.py). The objective
runs' own no-description curve stops at n = 4, so most of their ESS medians are only bounds; here the error with the
description under either prompt is read off the main run's whole no-description curve (n = 1 to 64), so that both
prompts' descriptions are priced in demonstrations of the same prompt without the sentence. Also the RMSE with the
description at n = 1, 2, 4 under each prompt. 95% bootstrap intervals and the paired difference, as analyze.main
draws them. Writes objective_ess_main_curve.csv.

    cd results/llm-test && python3 objective_main_curve.py       # then: python3 objective_main_curve.py tex
`tex` writes objective_table_main_curve.tex from objective_ess_main_curve.csv and objective_ess.csv: the share that is
a number, this ESS with its interval, and the weight at n = 1 and 4, with and without the sentence.
"""
import csv
import sys

import numpy as np

import analyze as A
import llm_test as T
from objective_analysis import BOOTS, CI, DIG, MODELS, NI, NONE, SDS, TASKS, pct, here



def tex():
    ess = {(r["model"], r["c"]): r for r in csv.DictReader(open(here / "objective_ess_main_curve.csv"))}
    first, obj = {}, {}
    for r in csv.DictReader(open(here / "objective_ess.csv")):
        if r["grid"] == "124" and first.setdefault((r["model"], r["prompt"]), r["readout"]) == r["readout"]:  # the first main readout listed
            obj[(r["model"], r["prompt"], r["c"])] = r
    f = lambda v: A.fmt_worth(float(v))
    cs = ["1", "4", "36"]
    lines = [r"\begin{tabular}{llcccccccccc}", r"\toprule",
             r"& & & \multicolumn{3}{c}{ESS, median over $n = 1$, $2$, $4$} & \multicolumn{3}{c}{weight, $n = 1$} & \multicolumn{3}{c}{weight, $n = 4$} \\",
             r"\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12}",
             r"Model & Prompt & number & $c = 1$ & $c = 4$ & $c = 36$ & $c = 1$ & $c = 4$ & $c = 36$ & $c = 1$ & $c = 4$ & $c = 36$ \\", r"\midrule",
             "Bayes-optimal & & & 1.3 & 4.4 & 36 & " + " & ".join(f"{c / (c + n):.2f}" for n in (1, 4) for c in (1, 4, 36)) + r" \\", r"\midrule"]
    for model in MODELS:
        for k, (p, name, key) in enumerate((("main", "main", ("ess_main", "main_lo", "main_hi")), ("objective", "loss stated", ("ess_objective", "obj_lo", "obj_hi")))):
            rs = {c: obj[(model, p, c)] for c in cs}
            share = np.mean([float(rs[c]["p_number"]) for c in cs])
            it = lambda s: rf"\textit{{{s}}}" if share < A.MIN_NUMBER else s
            cells = [it(f"{f(ess[(model, c)][key[0]])} {{\\scriptsize[{f(ess[(model, c)][key[1]])}, {f(ess[(model, c)][key[2]])}]}}") for c in cs]
            cells += [it(f"{float(rs[c][f'weight_n{n}']):.2f}") for n in (1, 4) for c in cs]
            nm = f"main ({first[(model, p)]})" if model == "Qwen3.5 9B" and p == "main" else name
            lines.append((model if k == 0 else "") + f" & {nm} & {share:.2f} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (here / "objective_table_main_curve.tex").write_text("\n".join(lines) + "\n")


if sys.argv[1:] == ["tex"]:
    tex()
    sys.exit()
rows = []
for model, (obj, mains) in MODELS.items():
    readout, path = mains[0]
    o, b = np.load(DIG / f"{obj}.npz")["mean"], np.load(path)["mean"]
    sq = lambda d, ci: (d[ci].astype(float) - TASKS["w"][:, None]) ** 2
    s0 = sq(b, NONE)  # the main run's no-description curve, every n
    med = lambda s1, i=slice(None): np.median([A.worth(np.nanmean(s0[i], 0), np.nanmean(s1[i], 0), n)[0] for n in A.MEDIAN_NS])
    for sd, ci in CI.items():
        so, sb = sq(o, ci), sq(b, ci)
        do, db = np.array([med(so, i) for i in BOOTS]), np.array([med(sb, i) for i in BOOTS])
        row = dict(model=model, main_readout=readout, c=round((T.MAIN / sd) ** 2), ess_objective=med(so), ess_main=med(sb))
        row["obj_lo"], row["obj_hi"] = pct(do)
        row["main_lo"], row["main_hi"] = pct(db)
        row["diff_lo"], row["diff_hi"] = pct(do - db)
        for n, k in zip(A.MEDIAN_NS, NI):
            row[f"rmse_objective_n{n}"], row[f"rmse_main_n{n}"] = np.sqrt(np.nanmean(so[:, k])), np.sqrt(np.nanmean(sb[:, k]))
        rows.append(row)
    print(model, "done", flush=True)
with open(here / "objective_ess_main_curve.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
for r in rows:
    print(f"{r['model']:12s} c={r['c']:2d}  objective {r['ess_objective']:.2f} [{r['obj_lo']:.2f}, {r['obj_hi']:.2f}]  main {r['ess_main']:.2f} [{r['main_lo']:.2f}, {r['main_hi']:.2f}]  "
          f"diff [{r['diff_lo']:.2f}, {r['diff_hi']:.2f}]  RMSE obj/main n=1 {r['rmse_objective_n1']:.1f}/{r['rmse_main_n1']:.1f} n=4 {r['rmse_objective_n4']:.1f}/{r['rmse_main_n4']:.1f}")
