"""Read the Luna replies: weight on the stated value and implied worth, against the Bayes-optimal values."""
import json, os, sys
import numpy as np
sys.path.insert(0, "..")
from llm_test import read_number  # the rule of every text reply: first number with three digits before any decimal part, decimals kept

rows = json.load(open("prompts.json"))
REPLIES = sys.argv[1] if len(sys.argv) > 1 else "replies"
pred = np.full(len(rows), np.nan)
for k in range(len(rows)):
    f = f"{REPLIES}/{k}.txt"
    if os.path.exists(f):
        pred[k] = read_number(open(f).read())

def fit(idx, boot=500):
    """Regress the prediction on the stated value and the mean of the numbers shown; return weight a, worth n*a/b."""
    X = np.array([[rows[k]["stated"], np.mean(rows[k]["shown"]), 1.0] for k in idx])
    y = pred[idx]
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    n = rows[idx[0]]["n"]
    rng = np.random.default_rng(0)
    def one(X, y):
        a, b, _ = np.linalg.lstsq(X, y, rcond=None)[0]
        return a, n * a / b
    a, w = one(X, y)
    bs = np.array([one(X[s], y[s]) for s in (rng.integers(0, len(y), len(y)) for _ in range(boot))])
    return a, w, bs.std(0), ok.mean(), len(y)

print(f"{'tol':>4} {'c':>3} {'n':>2} | {'weight':>7} {'ideal':>6} | {'worth':>6} {'ideal':>6} | {'rmse':>5} {'ideal':>5} | ok  N")
for tol in (60, 30, 10):
    c = (60 / tol) ** 2
    for n in (1, 4):
        idx = [k for k, r in enumerate(rows) if r["tol"] == tol and r["n"] == n]
        a, w, (sa, sw), comp, N = fit(idx)
        ok = np.isfinite(pred[idx])
        rmse = np.sqrt(np.mean((pred[idx][ok] - np.array([rows[k]["w"] for k in idx])[ok]) ** 2))
        irmse = np.sqrt(np.mean((np.array([rows[k]["ideal"] for k in idx]) - np.array([rows[k]["w"] for k in idx])) ** 2))
        print(f"{tol:>4} {c:>3.0f} {n:>2} | {a:>4.2f}±{sa:.2f} {c/(c+n):>6.2f} | {w:>6.1f} {c:>6.0f} | {rmse:>5.1f} {irmse:>5.1f} | {comp:.2f} {N}")


def table(passes=(("replies", "GPT-5.6-Luna, low reasoning"), ("replies-none", "GPT-5.6-Luna, reasoning off"))):
    """Write agent_table.tex: weight on the stated value and implied worth, both passes against the Bayes-optimal values."""
    global pred
    cells = [(tol, n) for n in (1, 4) for tol in (60, 30, 10)]
    out = ["\\begin{tabular}{lcccccccccccc}", "\\toprule",
           "& \\multicolumn{6}{c}{weight on the stated value} & \\multicolumn{6}{c}{implied worth, demonstrations} \\\\",
           "\\cmidrule(lr){2-7} \\cmidrule(lr){8-13}",
           "& \\multicolumn{3}{c}{$n = 1$} & \\multicolumn{3}{c}{$n = 4$} & \\multicolumn{3}{c}{$n = 1$} & \\multicolumn{3}{c}{$n = 4$} \\\\",
           "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13}",
           "Bayes-optimal ESS $c$ & 1 & 4 & 36 & 1 & 4 & 36 & 1 & 4 & 36 & 1 & 4 & 36 \\\\", "\\midrule"]
    bayes = ["Bayes-optimal"] + [f"{(60/t)**2/((60/t)**2+n):.2f}" for t, n in cells] + [f"{(60/t)**2:.0f}" for t, n in cells]
    out.append(" & ".join(bayes) + " \\\\")
    saved = pred.copy()
    for d, name in passes:
        pred[:] = np.nan
        for k in range(len(rows)):
            f = f"{d}/{k}.txt"
            if os.path.exists(f):
                pred[k] = read_number(open(f).read())
        ws, ts = [], []
        for t, n in cells:
            idx = [k for k, r in enumerate(rows) if r["tol"] == t and r["n"] == n]
            a, w, _, _, _ = fit(idx)
            ws.append(f"{a:.2f}"); ts.append(f"{w:.1f}")
        out.append(" & ".join([name] + ws + ts) + " \\\\")
    pred[:] = saved
    out += ["\\bottomrule", "\\end{tabular}"]
    open("agent_table.tex", "w").write("\n".join(out) + "\n")


if __name__ == "__main__" and os.path.isdir("replies-none"):
    table()


def ess_table(passes=(("replies", "replies_nodesc", "GPT-5.6-Luna, low reasoning"), ("replies-none", "replies_nodesc-none", "GPT-5.6-Luna, reasoning off")), boot=500):
    """Write agent_ess_table.tex: Definition 1 with squared error at n = 1 and 4 for c = 1, 4, 36, read from the
    model's own no-description curve (prompts_nodesc.json, n = 1 to 64, the same 100 tasks), interpolated in
    log n as in ../analyze.py; 95% bootstrap intervals over tasks. The Bayes-optimal row is
    read the same way from its predictions on the same tasks."""
    sys.path.insert(0, "..")
    import llm_test as T
    from analyze import worth, fmt_worth
    nd = json.load(open("prompts_nodesc.json"))
    NS, cells = (1, 2, 4, 8, 16, 32, 64), [(tol, n) for n in (1, 4) for tol in (60, 30, 10)]
    tasks = sorted({r["i"] for r in nd})
    def read(d, rows_):
        out = np.full(len(rows_), np.nan)
        for k in range(len(rows_)):
            f = f"{d}/{k}.txt"
            if os.path.exists(f):
                out[k] = read_number(open(f).read())
        return out
    def sq(p, rows_, key):  # (task, n) squared errors for the no-description prompts, or (task,) for one cell
        return {(r["i"], r["n"]): (p[k] - r["w"]) ** 2 for k, r in enumerate(rows_)}
    def curve(sqd, ns, idx):  # mean over the tasks idx at each n in ns, laid out on T.NS (nan where absent)
        mse = np.full(len(T.NS), np.nan)
        for n in ns:
            vals = [sqd[(i, n)] for i in idx if (i, n) in sqd]
            mse[T.NS.index(n)] = np.nanmean(vals) if vals else np.nan
        return mse
    rng = np.random.default_rng(0)
    boots = [tasks] + [list(rng.choice(tasks, len(tasks))) for _ in range(boot)]
    def ess_cells(pred_desc, pred_none):
        s_none = sq(pred_none, nd, "w")
        res = []
        for tol, n in cells:
            idx = [k for k, r in enumerate(rows) if r["tol"] == tol and r["n"] == n]
            s_desc = {(rows[k]["i"], n): (pred_desc[k] - rows[k]["w"]) ** 2 for k in idx}
            vals = [worth(curve(s_none, NS, b), curve(s_desc, (n,), b), n) for b in boots]
            (v, cens), bs = vals[0], np.array([x for x, _ in vals[1:]])
            lo, hi = np.percentile(bs, [2.5, 97.5], method="nearest")  # nearest: a draw may be inf (beyond the grid)
            cell = fmt_worth(v, cens, n, max(NS))
            res.append(cell if cens or not np.isfinite(v) else f"{cell} [{fmt_worth(lo, False, n, max(NS))}, {fmt_worth(hi, False, n, max(NS))}]")
        return res
    out = ["\\begin{tabular}{lcccccc}", "\\toprule",
           "& \\multicolumn{3}{c}{$n = 1$} & \\multicolumn{3}{c}{$n = 4$} \\\\", "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
           "Bayes-optimal ESS $c$ & 1 & 4 & 36 & 1 & 4 & 36 \\\\", "\\midrule"]
    ideal_desc, ideal_none = np.array([r["ideal"] for r in rows]), np.array([r["ideal"] for r in nd])
    out.append(" & ".join(["Bayes-optimal, read the same way"] + ess_cells(ideal_desc, ideal_none)) + " \\\\")
    for d, dn, name in passes:
        out.append(" & ".join([name] + ess_cells(read(d, rows), read(dn, nd))) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    open("agent_ess_table.tex", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__" and os.path.isdir("replies_nodesc-none"):
    ess_table()
