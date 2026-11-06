"""trust_slope.py: does each reply's implied trust track the reference's posterior relevance across tasks? (the paper's Appendix C.5)
Implied trust q = (pred - ybar)/(m - ybar) * (c + n)/c on tasks with |m - ybar| >= 30; reference pi the same from its
prediction. Updating predicts slope ~1 of q on pi; a constant-trust rule predicts slope 0."""
import sys, pathlib, numpy as np
H = pathlib.Path(__file__).parent
sys.path.insert(0, str(H.parent)); sys.path.insert(0, str(H))
import llm_test as T
import compare as C  # runs its table on import; fine
t, K = C.t, C.K
rng = np.random.default_rng(0)
for p in (0.5,):
    for name in C.NEW:
        for lab, d, idx in (("unstated", C.ORIG[name], lambda n, i: C.orig_index[(p, n)] * K + i),
                            ("stated", C.NEW[name], lambda n, i: C.CELLS.index((p, n)) * K + i)):
            Q, P, TK = [], [], []
            for n in (1, 2, 4):
                cond = (T.MAIN, 10, True, p)
                m, yb = T.stated(t, cond)[:K].astype(float), t["y"][:K, :n].mean(1)
                bayes = T.ideal_mean(t, cond, n, p)[:K]
                pred = np.array([C.number(d, idx(n, i)) for i in range(K)])
                ok = (np.abs(m - yb) >= 30) & ~np.isnan(pred)
                k = 36 / (36 + n)
                TK += list(np.nonzero(ok)[0])
                Q += list((pred[ok] - yb[ok]) / (m[ok] - yb[ok]) / k); P += list((bayes[ok] - yb[ok]) / (m[ok] - yb[ok]) / k)
            Q, P, TK = np.array(Q), np.array(P), np.array(TK)
            b = np.polyfit(P, Q, 1)[0]
            tasks = np.unique(TK)  # resample tasks, keeping each task's replies together
            rows = {k: np.nonzero(TK == k)[0] for k in tasks}
            bs = [np.polyfit(P[j], Q[j], 1)[0] for j in (np.concatenate([rows[k] for k in rng.choice(tasks, len(tasks))]) for _ in range(2000))]
            ic = np.polyfit(P, Q, 1)[1]; r2 = np.corrcoef(P, Q)[0, 1] ** 2
            low = P < 0.5  # tasks where the demonstrations argue against the description
            print(f"  intercept {ic:.2f}, R^2 {r2:.2f}; on {low.sum()} replies with reference trust < 0.5: mean implied trust {Q[low].mean():.2f} against the reference's {P[low].mean():.2f}")
            print(f"{name:18s} {lab:8s} p={p}: N={len(P)} replies on {len(tasks)} tasks; ref pi range {P.min():.2f}-{P.max():.2f} sd {P.std():.2f}; "
                  f"slope of implied trust on ref pi {b:.2f} [{np.percentile(bs,2.5):.2f}, {np.percentile(bs,97.5):.2f}]; "
                  f"corr {np.corrcoef(P,Q)[0,1]:.2f}")
