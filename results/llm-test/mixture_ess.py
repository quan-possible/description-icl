"""ESS under actual unreliability: the relevant and irrelevant conditions of the main run combined with weights
p and 1 - p (per task, then averaged), for a model that is not told the chance and for one that is, against the
fully trusting predictor, which takes the description as relevant, and the calibrated one, which knows p
and updates it from the demonstrations. Median over
MEDIAN_NS with a 95% bootstrap interval over tasks, from the model's own no-description curve.

    python3 mixture_ess.py full_gptoss20.npz ... written_gemma31.npz   # writes llm_table_mixture.tex
"""
import sys

import numpy as np

import llm_test as T
from analyze import BOOT, MEDIAN_NS, fmt_worth, here, model_label, worth

PS, TOLS = (0.9, 0.5), (30, 10)


def median_ess(sq_none, sq_desc, boots):
    point = [worth(np.nanmean(sq_none, 0), np.nanmean(sq_desc, 0), n) for n in MEDIAN_NS]
    draws = np.array([[worth(np.nanmean(sq_none[i], 0), np.nanmean(sq_desc[i], 0), n)[0] for n in MEDIAN_NS] for i in boots])
    med = np.median(draws, 1)
    return float(np.median([v for v, _ in point])), float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5)), any(c for _, c in point)


def cell(v, lo, hi, cens):
    s = fmt_worth(v, cens, max(MEDIAN_NS))
    return s if cens or not np.isfinite(v) else f"{s} [{fmt_worth(lo, False, max(MEDIAN_NS))}, {fmt_worth(hi, False, max(MEDIAN_NS))}]"


def main(paths):
    T.S = 2000
    t = T.make_tasks(T.MAIN)
    w = t["w"][:, None]
    ci = {cond: i for i, cond in enumerate(T.CONDITIONS)}
    ideal = lambda cond, p: (np.stack([T.ideal_mean(t, cond, n, p) for n in T.NS], 1) - w) ** 2
    boots = np.random.default_rng(0).integers(0, T.S, (BOOT, T.S))
    cols = [(p, tol) for p in PS for tol in TOLS]
    rows = []
    none = ideal((T.MAIN, None, True, None), 1.0)
    for name, p_trust in (("Fully trusting", lambda p: 1.0), ("Calibrated", lambda p: p)):
        cells = []
        for p, tol in cols:
            mix = p * ideal((T.MAIN, tol, True, None), p_trust(p)) + (1 - p) * ideal((T.MAIN, tol, False, None), p_trust(p))
            cells.append(cell(*median_ess(none, mix, boots)))
        rows.append((name, cells))
    for path in paths:
        d = np.load(path)
        label = model_label(str(d["repo"]))
        sq = lambda cond: (d["mean"][ci[cond]].astype(float) - w) ** 2
        none = sq((T.MAIN, None, True, None))
        for told, name in ((False, "not told"), (True, "told")):
            cells = []
            for p, tol in cols:
                q = p if told else None
                mix = p * sq((T.MAIN, tol, True, q)) + (1 - p) * sq((T.MAIN, tol, False, q))
                cells.append(cell(*median_ess(none, mix, boots)))
            rows.append((f"{label}, {name}", cells))
        print(path, "done", flush=True)
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"& \multicolumn{2}{c}{irrelevant one time in ten ($p = 0.9$)} & \multicolumn{2}{c}{irrelevant half the time ($p = 0.5$)} \\",
             r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
             r"Predictor & $c = 4$ & $c = 36$ & $c = 4$ & $c = 36$ \\", r"\midrule"]
    for k, (name, cells) in enumerate(rows):
        if k == 2:
            lines.append(r"\midrule")
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (here / "llm_table_mixture.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
