"""stated_all.py: every complete run of the stated-50% condition with the irrelevant case stated ("...in which case the
mean was instead drawn uniformly at random from the whole numbers 100 to 999"), scored as Figure 4 scores a panel:
summary ESS on the model's own baseline curve (median over n = 1, 2, 4, bootstrap over tasks), per-n ESS, and how
closely each reply follows the reference, which the prompt now fully determines, and fixed-weight averaging.
Haiku 5.5 low and Opus 4.6 without reasoning were stopped on 4 November after their 1,200 replies were in; Opus 4.6
high has 232 replies (stated 10% only) in replies-opus46-high and is not scored.

    cd results/llm-test/stated-fallback && python3 stated_all.py
"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import llm_test as T, analyze as A, reasoning as R

H = pathlib.Path(R.__file__).parent
RUNS = [("GPT-6 Sol, low", "codex", H / "sol-check/replies-gpt-6-sol-low-bare", H / "stated-fallback/replies-sol-low-bare"),
        ("Haiku 5.5, medium", "dir", H / "haiku/replies-medium", H / "stated-fallback/replies-haiku-medium"),
        ("Haiku 5.5, low", "dir", H / "haiku/replies-low", H / "stated-fallback/replies-haiku-low"),
        ("Opus 4.6, no thinking", "dir", H / "haiku/replies-opus46-none", H / "stated-fallback/replies-opus46-none")]

T.S = 2000
t = {k: v[: T.REASON_TASKS] for k, v in T.make_tasks(T.MAIN).items()}
T.S = K = T.REASON_TASKS
none = T.REASON_CONDS[0]
G = [0] + sorted(n for c, n in T.REASON_CELLS if c is none)
cell = {(c, n): k for k, (c, n) in enumerate(T.REASON_CELLS)}
new_rows = json.load(open(H / "stated-fallback/prompts.json"))
boots = np.random.default_rng(0).integers(0, K, (500, K))
p = 0.5
c = next(c for c in T.REASON_CONDS if c[1] == 10 and c[2] and c[3] == p)
ref = {n: T.ideal_mean(t, c, n, p) for n in (1, 2, 4)}
fixed = {n: p * T.ideal_mean(t, (T.MAIN, 10, True, None), n) + (1 - p) * T.ideal_mean(t, none, n) for n in (1, 2, 4)}
isq_none = {n: (T.ideal_mean(t, none, n) - t["w"]) ** 2 for n in G[1:]}


def summary(none_sq, desc_sq, idx):
    curve = np.array([np.nanmean(none_sq[n][idx]) if n else np.nan for n in G])
    per = [A.worth(curve, np.array([np.nanmean(desc_sq[n][idx]) if j == n else np.nan for j in G]), n, G)[0] for n in (1, 2, 4)]
    return float(np.median(per)), per


b, per = summary(isq_none, {n: (ref[n] - t["w"]) ** 2 for n in ref}, slice(None))
print(f"reference: ESS {b:.1f}  per n {np.round(per, 1)}")
b, per = summary(isq_none, {n: (fixed[n] - t["w"]) ** 2 for n in ref}, slice(None))
print(f"stated 50% never revised: ESS {b:.1f}  per n {np.round(per, 1)}\n")

for name, kind, orig, newdir in RUNS:
    pred, tokens = R.load(kind, orig)
    none_sq = {n: (pred[cell[none, n]] - t["w"]) ** 2 for n in G[1:]}
    for label in ("'unknown'", "spelled out"):
        got, think = {}, []
        for n in (1, 2, 4):
            if label == "'unknown'":
                got[n] = pred[cell[c, n]]
                continue
            v = np.full(K, np.nan)
            for j, r in enumerate(new_rows):
                if r["p"] == p and r["n"] == n:
                    f = newdir / (f"{j}.txt" if kind == "codex" else f"{j}.json")
                    if f.exists() and f.stat().st_size:
                        if kind == "codex":
                            v[r["task"]] = T.read_number(f.read_text())
                        else:
                            d = json.load(open(f)); v[r["task"]] = T.read_number(d["result"])
                            think.append(d["usage"]["output_tokens_details"]["thinking_tokens"] > 0)
            got[n] = v
        sq = {n: (got[n] - t["w"]) ** 2 for n in got}
        e, per = summary(none_sq, sq, slice(None))
        draws = [summary(none_sq, sq, i)[0] for i in boots]
        draws = [d if np.isfinite(d) else 1e9 for d in draws]
        allr = np.concatenate([got[n] for n in got]); allref = np.concatenate([ref[n] for n in got]); allfix = np.concatenate([fixed[n] for n in got])
        ok = np.isfinite(allr)
        sep = ok & (np.abs(allref - allfix) >= 5)
        closer = np.mean(np.abs(allr - allref)[sep] < np.abs(allr - allfix)[sep])
        # implied trust against the reference's (trust_slope.py): slope 1 if the model revises the stated chance as the reference does
        Q, P = [], []
        for n in got:
            m, yb = T.stated(t, c).astype(float), t["y"][:, :n].mean(1)
            use = (np.abs(m - yb) >= 30) & np.isfinite(got[n]); k = 36 / (36 + n)
            Q += list((got[n][use] - yb[use]) / (m[use] - yb[use]) / k); P += list((ref[n][use] - yb[use]) / (m[use] - yb[use]) / k)
        slope = np.polyfit(P, Q, 1)[0]
        print(f"{name:22s} {label:12s} answered {ok.mean():.3f}  ESS {e:5.1f} [{np.percentile(draws, 2.5):.1f}, {np.percentile(draws, 97.5):.1f}]  per n {np.round(per, 1)}  "
              f"median |reply - reference| {np.median(np.abs(allr - allref)[ok]):5.2f}, within 2: {np.mean(np.abs(allr - allref)[ok] <= 2):4.0%}; "
              f"vs never-revised {np.median(np.abs(allr - allfix)[ok]):5.2f}, {np.mean(np.abs(allr - allfix)[ok] <= 2):4.0%}; closer to reference {closer:4.0%} of {sep.sum()}; "
              f"trust slope {slope:.2f}" + (f"; reasoned {np.mean(think):.0%}" if think else ""))
    print()
