"""unstated_readings.py: at a stated 50% chance of being wrong under the original warning ("the mean is unknown"), which
reading of "unknown" do the models' replies with reasoning agree with? Readings = baseline priors for the mean when
the description is wrong, plus fixed-weight averaging (the 50% never updated). Prints each reading's summary ESS and,
per model, the share of replies within 2 of each reading's prediction. First 200 tasks, c = 36, n = 1, 2, 4.

    cd results/llm-test/stated-fallback && python3 unstated_readings.py
"""
import io, contextlib, pathlib, sys, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
with contextlib.redirect_stdout(io.StringIO()):
    import fallback_sensitivity as S
import llm_test as T, reasoning as R, analyze as A

K, m, t = S.K, S.m, {k: v[:S.K] for k, v in S.t.items()}
w = t["w"].astype(float)
cond = (60, 10, True, 0.5)
cells = list(T.REASON_CELLS); cell = {(c, n): k for k, (c, n) in enumerate(cells)}
none = T.REASON_CONDS[0]; G = [0] + sorted(n for c, n in cells if c is none)
READ = dict(S.FALLBACKS); READ["N(500, 150^2)"] = lambda y: S.gauss(y, np.full(len(y), 500.0), 150.0**2)
base = np.stack([(T.ideal_mean(t, none, n) - w) ** 2 for n in G], 1).mean(0)  # the paper reference's baseline curve

def predict(fb, n, q=0.5):
    y = t["y"][:K, :n].astype(float)
    mu1, l1 = S.gauss(y, m, S.tol**2)
    if fb is None:  # fixed rule: weights q and 1 - q, never revised; the wrong case predicts the mean of the numbers
        return q * mu1 + (1 - q) * y.mean(1), np.full(K, q), y.mean(1)
    mu0, l0 = fb(y)
    pi = 1 / (1 + (1 - q) / q * np.exp(np.clip(l0 - l1, -700, 700)))
    return pi * mu1 + (1 - pi) * mu0, pi, y.mean(1)

def ess(sq_by_n):
    vals = []
    for n in (1, 2, 4):
        at = np.full(len(G), np.nan); at[G.index(n)] = sq_by_n[n]
        vals.append(A.worth(base, at, n, G)[0])
    return np.median(vals), vals

cands = {"fixed rule (50% never revised)": None, **READ}
P = {name: {n: predict(fb, n) for n in (1, 2, 4)} for name, fb in cands.items()}
print("Where each candidate 'optimal answer' sits (summary ESS on the reference's baseline; per n):")
for name in cands:
    s, v = ess({n: ((P[name][n][0] - w) ** 2).mean() for n in (1, 2, 4)})
    print(f"  {name:34s} ESS {s:5.1f}  {np.round(v, 1)}   mean relevance prob at n=1,2,4: {[round(float(P[name][n][1].mean()), 2) for n in (1, 2, 4)]}  sd across tasks: {[round(float(P[name][n][1].std()), 2) for n in (1, 2, 4)]}")

runs = ["GPT-6 Sol, low", "Opus 4.6, high", "Gemma 4 31B, reasoning", "Qwen3.5 9B, reasoning", "Haiku 5.5, medium", "gpt-oss-20b, low"]
for run in runs:
    name_, kind, path = [r for r in R.RUNS if r[0] == run][0]
    pred, _ = R.load(kind, path)
    print(f"\n{run}")
    for name in cands:
        d = np.concatenate([np.abs(pred[cell[cond, n]] - P[name][n][0]) for n in (1, 2, 4)])
        # slope of the reply's normalized weight on this reading's relevance probability, tasks with |m - ybar| >= 30
        xs, ys = [], []
        for n in (1, 2, 4):
            _, pi, yb = P[name][n]; ok = (np.abs(m - yb) >= 30) & np.isfinite(pred[cell[cond, n]])
            xs.append(pi[ok]); ys.append(((pred[cell[cond, n]] - yb) / (m - yb))[ok] / (36 / (36 + n)))
        x, yv = np.concatenate(xs), np.concatenate(ys)
        slope = np.polyfit(x, yv, 1)[0] if x.std() > 1e-9 else float("nan")
        print(f"  vs {name:34s} median |reply - answer| {np.nanmedian(d):5.2f}   within 2: {np.nanmean(d <= 2):4.0%}   RMS {np.sqrt(np.nanmean(d**2)):6.2f}   slope on its relevance prob {slope:5.2f}")
