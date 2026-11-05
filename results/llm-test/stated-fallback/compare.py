"""compare.py: the paper's stated-reliability runs (fallback unstated: "in which case the mean is unknown") against this
rerun (fallback stated: uniform over 100..999), for GPT-6 Sol at low effort and Claude Haiku 5.5 at medium effort.

For each stated chance (10%, 50%) and n (1, 2, 4):
  - the weight on the stated value, from the paper's regression (stated value, mean of the numbers shown, constant);
  - the Bayes-optimal weight (llm_test.ideal_mean, which assumes the uniform fallback) and the fixed rule p * 36/(36 + n);
  - the share of replies closer to the Bayes-optimal prediction than to the fixed rule's, among tasks where the two
    differ by at least 5.
"""
import json
import pathlib
import sys

import numpy as np

H = pathlib.Path(__file__).parent
sys.path.insert(0, str(H.parent))
import llm_test as T

T.S = 2000
t = T.make_tasks(T.MAIN)
K = 200
ORIG = {"GPT-6 Sol, low": H.parent / "sol-check/replies-gpt-6-sol-low-bare", "Haiku 5.5, medium": H.parent / "haiku/replies-medium"}
NEW = {"GPT-6 Sol, low": H / "replies-sol-low-bare", "Haiku 5.5, medium": H / "replies-haiku-medium"}
CELLS = [(p, n) for p in (0.9, 0.5) for n in (1, 2, 4)]
orig_index = {(c[0][3], c[1]): k for k, c in enumerate(T.REASON_CELLS) if c[0][1] == 10 and c[0][2] and c[0][3]}


def number(d, k):
    for f in (d / f"{k}.txt", d / f"{k}.json"):
        if f.exists():
            txt = open(f).read()
            if f.suffix == ".json":
                txt = json.loads(txt)["result"]
            return T.read_number(txt)
    return np.nan


def weight(pred, m, ybar):
    ok = ~np.isnan(pred)
    X = np.column_stack([m[ok], ybar[ok], np.ones(ok.sum())])
    return np.linalg.lstsq(X, pred[ok], rcond=None)[0][0]


print(f"{'':20s} {'chance':>6s} {'n':>2s} {'Bayes':>6s} {'fixed':>6s} {'orig w':>7s} {'new w':>6s} {'orig->Bayes':>12s} {'new->Bayes':>11s}")
for name in NEW:
    for p, n in CELLS:
        cond = (T.MAIN, 10, True, p)
        m, ybar = T.stated(t, cond)[:K].astype(float), t["y"][:K, :n].mean(1)
        bayes = T.ideal_mean(t, cond, n, p)[:K]
        fixed_w = p * 36 / (36 + n)
        fixed = fixed_w * m + (1 - fixed_w) * ybar  # the fixed rule: the stated chance as a fixed weight
        out = []
        for d, idx in ((ORIG[name], lambda i: orig_index[(p, n)] * K + i), (NEW[name], lambda i: CELLS.index((p, n)) * K + i)):
            pred = np.array([number(d, idx(i)) for i in range(K)])
            diff = np.abs(bayes - fixed) >= 5
            ok = diff & ~np.isnan(pred)
            closer = np.mean(np.abs(pred[ok] - bayes[ok]) < np.abs(pred[ok] - fixed[ok])) if ok.any() else np.nan
            out.append((weight(pred, m, ybar), closer, ok.sum(), np.isnan(pred).sum()))
        bw = weight(bayes, m, ybar)
        print(f"{name:20s} {round(100 * (1 - p)):5d}% {n:2d} {bw:6.2f} {fixed_w:6.2f} {out[0][0]:7.2f} {out[1][0]:6.2f} "
              f"{out[0][1]:8.0%} ({out[0][2]:3d}) {out[1][1]:7.0%} ({out[1][2]:3d})" + (f"  missing {out[1][3]}" if out[1][3] else ""))
