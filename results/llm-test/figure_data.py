"""figure_data.py: the ESS behind the §6 figure, every model on the first 200 tasks of the main run.

    cd results/llm-test && python3 reasoning.py && python3 figure_data.py

Three conditions, as in the figure's panels: the relevant description at c = 36 (stated sd 10), the specificity
control (value accurate to sd 10, stated with sd 30), and the c = 36 description with a stated 50% chance of being
wrong. Runs with reasoning (and GPT-6 Sol and Opus 4.6 without it) come from reasoning.csv; the open models without
reasoning are scored here from their 2,000-task arrays restricted to the same 200 tasks, with the same ESS (Definition 1,
own no-description curve, median over n = 1, 2, 4) and the same bootstrap over tasks. Writes figure_data.csv."""
import csv
import pathlib

import numpy as np

import analyze as A
import llm_test as T

here = pathlib.Path(__file__).parent
K, BOOT = T.REASON_TASKS, 500
PANELS = {"c36": "relevant sd 10", "control": "relevant sd 30, value of sd 10", "wrong50": "relevant sd 10, 50% may be wrong"}
COND = {"c36": (60, 10, True, None), "control": (60, 30, True, None, 10), "wrong50": (60, 10, True, 0.5)}
# open models without reasoning: label -> (main array, control array); the paper reads gpt-oss-20b from its distribution
OPEN = {"gpt-oss-20b": ("full_gptoss20.npz", "control_gptoss20.npz"),
        "Qwen3.5 9B": ("written_qwen9.npz", "control_written_qwen9.npz"),
        "Gemma 4 31B": ("written_gemma31.npz", "control_written_gemma31.npz")}
# reasoning.csv run -> (model, mode)
RUNS = {"GPT-6 Sol, none": ("GPT-6 Sol", "without"), "GPT-6 Sol, low": ("GPT-6 Sol", "with"),
        "Opus 4.6, no thinking": ("Opus 4.6", "without"), "Opus 4.6, high": ("Opus 4.6", "with"),
        "Gemma 4 31B, reasoning": ("Gemma 4 31B", "with"), "Qwen3.5 9B, reasoning": ("Qwen3.5 9B", "with"),
        "gpt-oss-20b, low": ("gpt-oss-20b", "with"), "Haiku 5.5, medium": ("Haiku 5.5", "with")}


G = [0] + sorted(n for c, n in T.REASON_CELLS if c is T.REASON_CONDS[0])  # reasoning.py's grid (adds n = 40, 48)


def ess(none_sq, desc_sq, idx, grid=T.NS):
    """Median over n = 1, 2, 4 of Definition 1 on the tasks idx; none_sq (task, grid), desc_sq {n: (task,)}."""
    curve = np.nanmean(none_sq[idx], 0)
    vals = []
    for n in (1, 2, 4):
        at = np.full(len(grid), np.nan)
        at[list(grid).index(n)] = np.nanmean(desc_sq[n][idx])
        vals.append(A.worth(curve, at, n, list(grid))[0])
    return float(np.median(vals))


def open_rows():
    T.S = 2000
    t = {k: v[:K] for k, v in T.make_tasks(60).items()}
    boots = np.random.default_rng(0).integers(0, K, (BOOT, K))
    rows = []
    for model, (main, control) in OPEN.items():
        dm, dc = np.load(here / main), np.load(here / control)
        none_sq = (dm["mean"][0, :K].astype(float) - t["w"][:, None]) ** 2  # condition 0: no description, n on T.NS
        ideal_none = np.stack([(T.ideal_mean(t, T.CONDITIONS[0], n) - t["w"]) ** 2 for n in G], 1)  # Bayes on G, as in reasoning.py
        for panel, cond in COND.items():
            d, ci = (dc, T.CONTROL.index(cond)) if len(cond) == 5 else (dm, T.CONDITIONS.index(cond))
            desc = {n: (d["mean"][ci, :K, T.NS.index(n)].astype(float) - t["w"]) ** 2 for n in (1, 2, 4)}
            ideal = {n: (T.ideal_mean(t, cond, n, cond[3] or 1.0) - t["w"]) ** 2 for n in (1, 2, 4)}
            all_ = np.arange(K)
            draws, ref_draws = [ess(none_sq, desc, b) for b in boots], [ess(ideal_none, desc, b, G) for b in boots]
            rows.append(dict(model=model, mode="without", panel=panel, ess=ess(none_sq, desc, all_),
                             lo=np.percentile(draws, 2.5), hi=np.percentile(draws, 97.5),
                             ref_ess=ess(ideal_none, desc, all_, G), ref_lo=np.percentile(ref_draws, 2.5),
                             ref_hi=np.percentile(ref_draws, 97.5), bayes=ess(ideal_none, ideal, all_, G)))
    return rows


def reasoning_rows():
    rows = []
    for r in csv.DictReader(open(here / "reasoning.csv")):
        if r["run"] in RUNS:
            for panel, name in PANELS.items():
                if r["condition"] == name:
                    model, mode = RUNS[r["run"]]
                    rows.append(dict(model=model, mode=mode, panel=panel, ess=r["ess"], lo=r["ess_lo"], hi=r["ess_hi"],
                                     ref_ess=r["ref_ess"], ref_lo=r["ref_ess_lo"], ref_hi=r["ref_ess_hi"], bayes=r["ideal_ess"]))
    return rows


if __name__ == "__main__":
    rows = open_rows() + reasoning_rows()
    with open(here / "figure_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (f"{float(v):.4g}" if k not in ("model", "mode", "panel") else v) for k, v in r.items()} for r in rows)
    for r in rows:
        print(f"{r['model']:12s} {r['mode']:8s} {r['panel']:8s} ESS {float(r['ess']):6.1f} [{float(r['lo']):.1f}, {float(r['hi']):.1f}]  ref {float(r['ref_ess']):5.1f}  Bayes {float(r['bayes']):.1f}")
