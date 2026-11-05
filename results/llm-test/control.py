"""The specificity control (CONTROL in llm_test.py): with the stated value and the numbers fixed, does a model's weight on
the stated value, and the description's ESS, follow the stated standard deviation?

    python3 control.py control_gptoss20.npz control_phi4.npz control_olmo32.npz control_written_{gptoss20,qwen9,gemma12,gemma31}.npz

writes control.csv (every model, actual accuracy, stated sd and n) and prints the headline: the weight at n = 1 under each
stated sd, the change in weight from stated sd 60 to 10 with a 95% bootstrap interval over tasks, and the ESS (median over
n = 1, 2, 4, read from the run's own no-description curve as in analyze.py), each against the Bayes-optimal predictor that
takes the statement at face value.
"""

import csv
import pathlib
import sys

import numpy as np

import analyze as A
import llm_test as T

here = pathlib.Path(__file__).parent
BOOT = 500


def main(paths):
    T.S = 2000
    t = T.make_tasks(T.MAIN)
    boots = np.random.default_rng(0).integers(0, T.S, (BOOT, T.S))
    rows, rows_contrast = [], []
    for path in paths:
        d = np.load(path)
        model = A.model_label(str(d["repo"]).replace(" (control)", ""))
        pred = {cond: d["mean"][ci].astype(float) for ci, cond in enumerate(T.CONTROL)}  # (task, n)
        ideal = {cond: np.stack([T.ideal_mean(t, cond, n) for n in T.NS], 1) for cond in T.CONTROL}
        none = T.CONTROL[0]
        main = here / pathlib.Path(path).name.replace("control_", "" if "written" in path or "digits" in path else "full_")
        if main.exists():  # the control's no-description and matched cells repeat prompts of the main run
            M = np.load(main)["mean"].astype(float)
            for cond in (none, (T.MAIN, 60, True, None, 60), (T.MAIN, 10, True, None, 10)):
                same = T.CONDITIONS.index(cond[:4])
                print(f"{model:24s} reproduces {T.label(cond)}: median |difference| {np.nanmedian(abs(pred[cond] - M[same])):.3f}, "
                      f"share within 1: {np.nanmean(abs(pred[cond] - M[same]) < 1):.3f}")
        sq = lambda p: (p - t["w"][:, None]) ** 2
        for cond in T.CONTROL[1:]:
            m = T.stated(t, cond).astype(float)
            for ni, n in enumerate(T.NS):
                if n == 0:
                    continue
                ybar, p, q = t["y"][:, :n].mean(1), pred[cond][:, ni], ideal[cond][:, ni]
                keep = np.isfinite(p)  # as in analyze.py: a reply without a number is left out
                w = A.fit(p[keep], m[keep], ybar[keep])[0]
                wb = np.array([A.fit(p[i], m[i], ybar[i])[0] for i in (j[keep[j]] for j in boots)])
                rows.append(dict(model=model, actual=cond[4], stated=cond[1], n=n, number=float(keep.mean()),
                                 weight=w, weight_lo=np.percentile(wb, 2.5), weight_hi=np.percentile(wb, 97.5), ideal_weight=A.fit(q[keep], m[keep], ybar[keep])[0],
                                 ess=A.worth(np.nanmean(sq(pred[none]), 0), np.nanmean(sq(pred[cond]), 0), n)[0],
                                 ideal_ess=A.worth(sq(ideal[none]).mean(0), sq(ideal[cond]).mean(0), n)[0],
                                 error=float(np.sqrt(np.mean((p - t["w"])[keep] ** 2))), ideal_error=float(np.sqrt(np.mean((q - t["w"])[keep] ** 2)))))
        for acc in (10, 60):
            # the primary contrast: the paired change in prediction from stated sd 60 to 10 (value and numbers fixed),
            # regressed across tasks and n = 1, 2, 4 on the face-value Bayes-optimal change; 1 = follows the stated sd
            # as the Bayes-optimal predictor does, 0 = ignores it. On tasks answered in both cells.
            lo, hi = (T.MAIN, 60, True, None, acc), (T.MAIN, 10, True, None, acc)
            cols = [T.NS.index(n) for n in A.MEDIAN_NS]
            dm, di = pred[hi][:, cols] - pred[lo][:, cols], ideal[hi][:, cols] - ideal[lo][:, cols]  # (task, n)

            def slope(i):  # pooled over n, on the (resampled) tasks i, leaving out pairs with a missing answer
                x, y = di[i].ravel(), dm[i].ravel()
                keep = np.isfinite(y)
                return np.polyfit(x[keep], y[keep], 1)[0]

            ok = np.isfinite(dm)
            draws = [slope(i) for i in boots]
            print(f"{model:24s} value of sd {acc}: response to the stated sd, share of the Bayes-optimal {slope(slice(None)):.2f} "
                  f"[{np.percentile(draws, 2.5):.2f}, {np.percentile(draws, 97.5):.2f}]  (tasks answered in both: {ok.mean():.3f})")
            rows_contrast.append(dict(model=model, actual=acc, share=slope(slice(None)), lo=np.percentile(draws, 2.5), hi=np.percentile(draws, 97.5)))
        for acc in (10, 60):  # the change in weight at n = 1 from stated sd 60 to 10, with the value and numbers fixed
            lo, hi = (T.MAIN, 60, True, None, acc), (T.MAIN, 10, True, None, acc)
            m, ybar = T.stated(t, lo).astype(float), t["y"][:, 0]
            def f(cond, i):  # the weight on the tasks i that have an answer
                i = np.arange(T.S)[i]
                i = i[np.isfinite(pred[cond][i, 1])]
                return A.fit(pred[cond][i, 1], m[i], ybar[i])[0]
            diff = np.array([f(hi, i) - f(lo, i) for i in boots])
            ideal_diff = A.fit(ideal[hi][:, 1], m, ybar)[0] - A.fit(ideal[lo][:, 1], m, ybar)[0]
            print(f"{model:24s} value of sd {acc}: weight change, stated 60 -> 10, n = 1: {f(hi, slice(None)) - f(lo, slice(None)):+.3f} "
                  f"[{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}]  Bayes-optimal {ideal_diff:+.3f}")
    with open(here / "control.csv", "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    for r in rows:
        if r["n"] == 1:
            print(f"{r['model']:24s} value of sd {r['actual']:2d}, stated {r['stated']:2d}: weight {r['weight']:.2f} ({r['ideal_weight']:.2f})  "
                  f"ESS {r['ess']:6.1f} ({r['ideal_ess']:5.1f})  number {r['number']:.2f}")
    for model in dict.fromkeys(r["model"] for r in rows):  # the headline ESS: median over n = 1, 2, 4
        for acc in (10, 60):
            cells = [(s, [r for r in rows if r["model"] == model and r["actual"] == acc and r["stated"] == s and r["n"] in A.MEDIAN_NS]) for s in (60, 30, 10)]
            print(f"{model:24s} value of sd {acc}: median ESS stated 60/30/10 " + "  ".join(f"{np.median([r['ess'] for r in rs]):.1f} ({np.median([r['ideal_ess'] for r in rs]):.1f})" for s, rs in cells))
    with open(here / "control_contrast.csv", "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows_contrast[0]))
        out.writeheader()
        out.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows_contrast)
    return rows


def table():
    """control_table.tex from control.csv (open models, 2,000 tasks): for the value accurate to sd 10, the weight at
    n = 1 and the ESS (median over n = 1, 2, 4) under each stated sd. The reasoning runs' control is in reason_table.tex."""
    rows = list(csv.DictReader(open(here / "control.csv")))
    f2 = lambda v: ("0.00" if abs(float(v)) < 0.005 else f"{float(v):.2f}").replace("-", "$-$")
    ess = lambda v: A.fmt_worth(float(v), False, 4)
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r"& \multicolumn{3}{c}{weight, $n = 1$} & \multicolumn{3}{c}{ESS} \\",
           r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
           r"stated $c$ & $1$ & $4$ & $36$ & $1$ & $4$ & $36$ \\", r"\midrule"]
    models = list(dict.fromkeys(r["model"] for r in rows))
    pick = lambda m, s_, n: next(r for r in rows if r["model"] == m and int(r["actual"]) == 10 and int(r["stated"]) == s_ and int(r["n"]) == n)
    med = lambda m, s_, key: float(np.median([float(pick(m, s_, n)[key]) for n in A.MEDIAN_NS]))
    out.append("Bayes-optimal & " + " & ".join(f2(pick(models[0], s_, 1)["ideal_weight"]) for s_ in (60, 30, 10)) + " & " + " & ".join(ess(med(models[0], s_, "ideal_ess")) for s_ in (60, 30, 10)) + r" \\")
    out.append(r"\midrule")
    for m in models:
        out.append(f"{m} & " + " & ".join(f2(pick(m, s_, 1)["weight"]) for s_ in (60, 30, 10)) + " & " + " & ".join(ess(med(m, s_, "ess")) for s_ in (60, 30, 10)) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (here / "control_table.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
    table()
