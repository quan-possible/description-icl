"""ess.py: the summary ESS (median over n = 1, 2, 4) of the c = 36 description with a stated chance of being wrong, for
the rerun with the fallback stated, computed exactly as reasoning.py computes Table 8 and Figure 4(c): each model's own
no-description curve from its original run, the description's errors from the rerun."""
import json, pathlib, sys
import numpy as np
H = pathlib.Path(__file__).parent; sys.path.insert(0, str(H.parent))
import llm_test as T, analyze as A, reasoning as R

T.S = 2000
t = {k: v[: T.REASON_TASKS] for k, v in T.make_tasks(T.MAIN).items()}
T.S = T.REASON_TASKS
none = T.REASON_CONDS[0]
G = [0] + sorted(n for c, n in T.REASON_CELLS if c is none)
cell = {(c, n): k for k, (c, n) in enumerate(T.REASON_CELLS)}
new_rows = json.load(open(H / "prompts.json"))
for name, kind, orig, newdir in [("GPT-6 Sol, low", "codex", H.parent / "sol-check/replies-gpt-6-sol-low", H / "replies-sol-low"),
                                 ("Haiku 5.5, medium", "dir", H.parent / "haiku/replies-medium", H / "replies-haiku-medium")]:
    pred, _ = R.load(kind, orig)
    sq_orig = {key: (pred[k] - t["w"]) ** 2 for key, k in cell.items()}
    for p in (0.9, 0.5):
        c = next(c for c in T.REASON_CONDS if c[1] == 10 and c[2] and c[3] == p)
        out = {}
        for label, src in (("original", None), ("fallback stated", newdir)):
            sq = dict(sq_orig)
            if src is not None:
                for n in (1, 2, 4):
                    v = np.full(T.REASON_TASKS, np.nan)
                    for j, r in enumerate(new_rows):
                        if r["p"] == p and r["n"] == n:
                            f = src / (f"{j}.txt" if kind == "codex" else f"{j}.json")
                            if f.exists():
                                txt = f.read_text() if kind == "codex" else json.load(open(f))["result"]
                                v[r["task"]] = T.read_number(txt)
                    sq[c, n] = (v - t["w"]) ** 2
            curve = np.array([np.nanmean(sq[none, n]) if n else np.nan for n in G])
            es = [A.worth(curve, np.array([np.nanmean(sq[c, n]) if j == n else np.nan for j in G]), n, G)[0] for n in (1, 2, 4)]
            out[label] = float(np.median(es))
        ideal = {n: (T.ideal_mean(t, c, n, p) - t["w"]) ** 2 for n in (1, 2, 4)}
        isq_curve = np.array([np.nanmean((T.ideal_mean(t, none, n) - t["w"]) ** 2) if n else np.nan for n in G])
        bay = float(np.median([A.worth(isq_curve, np.array([np.nanmean(ideal[n]) if j == n else np.nan for j in G]), n, G)[0] for n in (1, 2, 4)]))
        print(f"{name:18s} stated {round(100 * (1 - p)):2d}%: ESS original {out['original']:5.1f}, fallback stated {out['fallback stated']:5.1f}, Bayes-optimal {bay:5.1f}")
