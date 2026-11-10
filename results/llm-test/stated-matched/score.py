"""score.py: the matched stated-reliability design, scored. Every one of the 200 main tasks is run with its relevant
c = 36 description (../stated-fallback) and with an irrelevant one (this folder), under a stated 10% or 50% chance of
being wrong with the irrelevant case stated. The error with the description is the two arms' mean squared errors
weighted by the stated chance, so the stated chance is also the true one. ESS is Definition 1 against each model's own
curve without a description from its main run, summarized as the median over n = 1, 2, 4, with a 95% task-bootstrap
interval; a value of 1 - n is censored (the description is worse than one demonstration alone). Also prints, per arm,
the share of replies within 2 of the reference's prediction and of the fixed-weight prediction."""
import json, pathlib, sys
import numpy as np
H = pathlib.Path(__file__).parent; sys.path.insert(0, str(H.parent)); sys.path.insert(0, str(H))
import llm_test as T, analyze as A, reasoning as R, exact as E

P = H.parent
MODELS = [("GPT-6 Sol, low", "codex", P / "sol-check/replies-gpt-6-sol-low-bare", P / "stated-fallback/replies-sol-low-bare", H / "replies-sol-low"),
          ("Haiku 5.5, medium", "dir", P / "haiku/replies-medium", P / "stated-fallback/replies-haiku-medium", H / "replies-haiku-medium"),
          ("GPT-6 Sol, none", "codex", P / "sol-check/replies-gpt-6-sol-none-bare", P / "stated-fallback/replies-sol-none-bare", H / "replies-sol-none"),
          ("Opus 4.6, high", "dir", P / "haiku/replies-opus46-high", P / "stated-fallback/replies-opus46-high", H / "replies-opus46-high"),
          ("Opus 4.6, none", "dir", P / "haiku/replies-opus46-none", P / "stated-fallback/replies-opus46-none", H / "replies-opus46-none")]
t = {k: v[:200] for k, v in E.full.items()}
T.S = 200
none = T.REASON_CONDS[0]
G = [0] + sorted(n for c, n in T.REASON_CELLS if c is none)
cellof = {(c, n): k for k, (c, n) in enumerate(T.REASON_CELLS)}
rows = json.load(open(H / "prompts.json"))
NS = (1, 2, 4)


def arm(kind, d):
    """{(p, n): predictions over the 200 tasks} from a reply folder indexed like prompts.json."""
    out = {(p, n): np.full(200, np.nan) for p in (0.9, 0.5) for n in NS}
    for j, r in enumerate(rows):
        f = d / (f"{j}.txt" if kind == "codex" else f"{j}.json")
        if f.exists() and f.stat().st_size:
            out[r["p"], r["n"]][r["task"]] = T.read_number(f.read_text() if kind == "codex" else json.load(open(f))["result"])
    return out


def summary(curve_sq, rel_sq, irr_sq, p, idx):
    curve = np.array([np.nanmean(curve_sq[n][idx]) if n else np.nan for n in G])
    es = []
    for n in NS:
        r = p * np.nanmean(rel_sq[n][idx]) + (1 - p) * np.nanmean(irr_sq[n][idx])
        es.append(A.worth(curve, np.array([r if j == n else np.nan for j in G]), n, G)[0])
    return es


def rule_preds(p, n, col, rule):
    return E.preds(t, t[col], p, n, rule)


rng = np.random.default_rng(0)
ALL = np.arange(200)
OUT = []  # rows of matched.csv, read by ../figure_data.py
for p in (0.9, 0.5):
    print(f"\n=== stated and true chance of irrelevance {round(100 * (1 - p))}% ===")
    base = {n: (T.ideal_mean(t, none, n) - t["w"]) ** 2 if n else np.full(200, np.nan) for n in G}
    for rule in ("reference", "fixed weights", "ignores warning"):
        rel = {n: (rule_preds(p, n, "valid10", rule) - t["w"]) ** 2 for n in NS}
        irr = {n: (rule_preds(p, n, "invalid10", rule) - t["w"]) ** 2 for n in NS}
        e = summary(base, rel, irr, p, ALL)
        b = [np.median(summary(base, rel, irr, p, rng.integers(0, 200, 200))) for _ in range(1000)]
        print(f"{rule:18s} per n {e[0]:5.1f} {e[1]:5.1f} {e[2]:5.1f}  summary {np.median(e):5.1f} [{np.percentile(b, 2.5):.1f}, {np.percentile(b, 97.5):.1f}]")
        OUT.append(dict(run=rule, stated=round(100 * (1 - p)), ess=np.median(e), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), n1=e[0], n2=e[1], n4=e[2], rel_ref="", irr_ref="", rel_fixed="", irr_fixed=""))
    for name, kind, orig, reldir, irrdir in MODELS:
        if not (reldir.exists() and irrdir.exists() and orig.exists()):
            continue
        pred, _ = R.load(kind, orig)
        base = {n: (pred[cellof[none, n]] - t["w"]) ** 2 if n else np.full(200, np.nan) for n in G}
        a_rel, a_irr = arm(kind, reldir), arm(kind, irrdir)
        got = [int(np.isfinite(a[p, n]).sum()) for a in (a_rel, a_irr) for n in NS]
        if min(got) < 150:
            print(f"{name:18s} incomplete: replies with a number per cell {got}")
            continue
        rel = {n: (a_rel[p, n] - t["w"]) ** 2 for n in NS}
        irr = {n: (a_irr[p, n] - t["w"]) ** 2 for n in NS}
        e = summary(base, rel, irr, p, ALL)
        b = [np.median(summary(base, rel, irr, p, rng.integers(0, 200, 200))) for _ in range(1000)]
        share = {}
        for lab, a, col in (("rel", a_rel, "valid10"), ("irr", a_irr, "invalid10")):
            for rule in ("reference", "fixed weights"):
                d = np.concatenate([np.abs(a[p, n] - rule_preds(p, n, col, rule)) for n in NS])
                share[lab, rule] = np.mean(d[np.isfinite(d)] <= 2)
        print(f"{name:18s} per n {e[0]:5.1f} {e[1]:5.1f} {e[2]:5.1f}  summary {np.median(e):5.1f} [{np.percentile(b, 2.5):.1f}, {np.percentile(b, 97.5):.1f}]"
              f"  within 2 of reference: relevant {share['rel', 'reference']:.0%}, irrelevant {share['irr', 'reference']:.0%};"
              f" of fixed weights: {share['rel', 'fixed weights']:.0%}, {share['irr', 'fixed weights']:.0%}  (numbers per cell {min(got)}-{max(got)})")
        OUT.append(dict(run=name, stated=round(100 * (1 - p)), ess=np.median(e), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), n1=e[0], n2=e[1], n4=e[2],
                        rel_ref=share["rel", "reference"], irr_ref=share["irr", "reference"], rel_fixed=share["rel", "fixed weights"], irr_fixed=share["irr", "fixed weights"]))

import csv
with open(H / "matched.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(OUT[0])); w.writeheader()
    w.writerows({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()} for r in OUT)
