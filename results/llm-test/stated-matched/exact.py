"""exact.py: what the prompt-based Bayesian reference and two simpler rules score when the stated chance of
irrelevance is also the true one. Each task's c = 36 description is relevant with probability p (its `valid10`
value) and otherwise an independently drawn value (`invalid10`); the prompt states the chance and the irrelevant
case (uniform over 100..999). Compares with the paper's design, where every description is relevant.
Prints summary ESS (median over n = 1, 2, 4) on the 200 main tasks, with task-bootstrap intervals, and on 2,000."""
import pathlib, sys
import numpy as np
H = pathlib.Path(__file__).parent; sys.path.insert(0, str(H.parent))
import llm_test as T, analyze as A

T.S = 2000
full = T.make_tasks(T.MAIN)
none = (T.MAIN, None, True, None)
G = [0, 1, 2, 4, 8, 16, 32, 40, 48, 64]


def relevance(p, size, seed=1):
    """Which tasks get a relevant description: an exact share p, in a seeded random order."""
    z = np.zeros(size, bool); z[: round(p * size)] = True
    return np.random.default_rng([seed, round(100 * p)]).permutation(z)


def preds(t, m, p, n, rule):
    tt = dict(t); tt["valid10"] = m  # the value shown, whatever its origin
    cond = (T.MAIN, 10, True, p)
    if rule == "reference":
        return T.ideal_mean(tt, cond, n, p)
    if rule == "ignores warning":
        return T.ideal_mean(tt, cond, n, 1.0)
    if rule == "fixed weights":
        return p * T.ideal_mean(tt, cond, n, 1.0) + (1 - p) * T.ideal_mean(tt, none, n)


def ess(t, m, p, rule, idx=None):
    idx = np.arange(len(t["w"])) if idx is None else idx
    sub = {k: v[idx] for k, v in t.items()}
    curve = np.array([np.mean((T.ideal_mean(sub, none, n) - sub["w"]) ** 2) if n else np.nan for n in G])
    out = []
    for n in (1, 2, 4):
        r = np.mean((preds(sub, m[idx], p, n, rule) - sub["w"]) ** 2)
        out.append(A.worth(curve, np.array([r if j == n else np.nan for j in G]), n, G)[0])
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for size in (200, 2000):
        t = {k: v[:size] for k, v in full.items()}
        for p in (0.5, 0.9):
            z = relevance(p, size)
            for design, m in (("all relevant", t["valid10"]), ("matched", np.where(z, t["valid10"], t["invalid10"]))):
                for rule in ("reference", "fixed weights", "ignores warning"):
                    e = ess(t, m, p, rule)
                    line = f"{size:5d} tasks  stated {round(100 * (1 - p)):2d}%  {design:12s}  {rule:16s}  per n {e[0]:6.1f} {e[1]:6.1f} {e[2]:6.1f}  summary {np.median(e):6.1f}"
                    if size == 200:
                        b = [np.median(ess(t, m, p, rule, rng.integers(0, size, size))) for _ in range(400)]
                        line += f"  [{np.nanpercentile(b, 2.5):.1f}, {np.nanpercentile(b, 97.5):.1f}]"
                    print(line)
        print()


def ess_paired(t, p, rule, idx=None):
    """Every task run with its relevant and with its irrelevant description; risks weighted p and 1 - p."""
    idx = np.arange(len(t["w"])) if idx is None else idx
    sub = {k: v[idx] for k, v in t.items()}
    curve = np.array([np.mean((T.ideal_mean(sub, none, n) - sub["w"]) ** 2) if n else np.nan for n in G])
    out = []
    for n in (1, 2, 4):
        r = sum(wt * np.mean((preds(sub, sub[col], p, n, rule) - sub["w"]) ** 2) for wt, col in ((p, "valid10"), (1 - p, "invalid10")))
        out.append(A.worth(curve, np.array([r if j == n else np.nan for j in G]), n, G)[0])
    return out


if __name__ == "__main__":
    print("paired: every task with both descriptions, risks weighted by the stated chance")
    t = {k: v[:200] for k, v in full.items()}
    for p in (0.5, 0.9, 1.0):
        for rule in ("reference", "fixed weights", "ignores warning"):
            if p == 1.0 and rule != "reference":
                continue
            e = ess_paired(t, p, rule)
            b = [np.median(ess_paired(t, p, rule, rng.integers(0, 200, 200))) for _ in range(400)]
            print(f"  200 tasks  stated {round(100 * (1 - p)):2d}%  {rule:16s}  per n {e[0]:6.1f} {e[1]:6.1f} {e[2]:6.1f}  summary {np.median(e):6.1f}  [{np.nanpercentile(b, 2.5):.1f}, {np.nanpercentile(b, 97.5):.1f}]")
