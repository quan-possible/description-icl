"""The reasoning runs (REASON_CELLS of llm_test.py, the first 200 of the main run's 2,000 tasks): gpt-oss-20b reasoning
at low effort (vLLM, 4,096 tokens) and Claude Haiku 5.5 through the Claude Code CLI at efforts medium and low.

    python3 reasoning.py      # reads reason_gptoss20_4096.npz and haiku/replies-{medium,low}/ where present

For each run and description condition: the share of replies with a number, the weight on the stated value at n = 1
and 4 (regression on the stated value, the shown mean and a constant), the error, and the ESS (Definition 1 with squared
error, read from the run's own no-description curve at n = 1, 2, ..., 64 as in analyze.py; median over n = 1, 2, 4,
with a 95% bootstrap interval over tasks), each against the Bayes-optimal predictor on the same tasks. A reply without a
number is left out of the error curves, as in analyze.py. Writes reasoning.csv and reason_table.tex.
"""

import csv
import json
import os
import pathlib

import numpy as np

import analyze as A
import llm_test as T

here = pathlib.Path(__file__).parent
BOOT = 500
RUNS = [("gpt-oss-20b, low", "npz", here / "reason_gptoss20_4096.npz"),
        ("Haiku 5.5, medium", "dir", here / "haiku/replies-medium"),
        ("Opus 4.6, high", "dir", here / "haiku/replies-opus46-high"),
        ("Opus 4.6, no thinking", "dir", here / "haiku/replies-opus46-none"),
        ("Haiku 5.5, low", "dir", here / "haiku/replies-low"),
        ("GPT-6 Sol, low", "codex", here / "sol-check/replies-gpt-6-sol-low-bare"),
        ("GPT-6 Sol, none", "codex", here / "sol-check/replies-gpt-6-sol-none-bare"),
        ("Qwen3.5 9B, reasoning", "openrouter", here / "openrouter/qwen35-think"),
        ("Gemma 4 31B, reasoning", "openrouter", here / "openrouter/gemma31-think"),  # Gemini API, openrouter/gemini.py
        ("Olmo 3 7B Think", "olmo", here / "sol-check/olmo3-think-full.json")]


def load(kind, path):
    """pred[cell, task] (nan where no number or cell not run) and the generated (gpt-oss) or thinking (Haiku) tokens."""
    shape = (len(T.REASON_CELLS), T.REASON_TASKS)
    pred, tokens = np.full(shape, np.nan), np.full(shape, np.nan)
    if kind == "npz":
        d = np.load(path)
        pred[: len(d["pred"])], tokens[: len(d["pred"])] = d["pred"], d["tokens"]  # runs before the n = 40, 48 cells lack them
        return pred, tokens
    prompts = json.load(open(here / "haiku/prompts.json"))
    if kind == "olmo":  # {prompt index: [answer, tokens]} from sol-check/olmo_control.py
        for j, (ans, tok) in json.load(open(path)).items():
            q = prompts[int(j)]
            pred[q["cell"], q["task"]], tokens[q["cell"], q["task"]] = T.read_number(ans or ""), tok
        return pred, tokens
    for j, q in enumerate(prompts):
        f, g = path / f"{j}.json", path / f"{j}.txt"
        if kind == "dir" and f.exists():  # Claude Code CLI records (Haiku)
            r = json.load(open(f))
            pred[q["cell"], q["task"]] = T.read_number(r["result"])
            tokens[q["cell"], q["task"]] = r["usage"]["output_tokens_details"]["thinking_tokens"]
        elif kind == "openrouter" and f.exists():  # openrouter/run.py records
            r = json.load(open(f))
            rs = path.parent / f"{path.name}-resampled/{j}.json"  # cut off at the token limit: resampled once (run.py --sample)
            if r["finish_reason"] == "length" and rs.exists():
                r = json.load(open(rs))
            pred[q["cell"], q["task"]], tokens[q["cell"], q["task"]] = T.read_number(r["answer"]), r["reasoning_tokens"] or 0
        elif kind == "codex" and g.exists() and g.stat().st_size:  # sol-check/ask.sh replies (bare text)
            log = g.with_suffix(".jsonl")  # SKIP_TOOL_REPLIES=1 leaves out replies whose answer Sol computed in a shell
            if os.environ.get("SKIP_TOOL_REPLIES") and log.exists() and '"command_execution"' in log.read_text(errors="ignore"):
                continue
            pred[q["cell"], q["task"]] = T.read_number(g.read_text())
    return pred, tokens


def main():
    T.S = 2000
    t = {k: v[: T.REASON_TASKS] for k, v in T.make_tasks(T.MAIN).items()}
    T.S = T.REASON_TASKS
    boots = np.random.default_rng(0).integers(0, T.REASON_TASKS, (BOOT, T.REASON_TASKS))
    none = T.REASON_CONDS[0]
    G = [0] + sorted(n for c, n in T.REASON_CELLS if c is none)  # the no-description curve's numbers shown
    conds = list(dict.fromkeys(c for c, _ in T.REASON_CELLS if c is not none))
    cell = {(c, n): k for k, (c, n) in enumerate(T.REASON_CELLS)}
    ideal = {(c, n): T.ideal_mean(t, c, n, c[3] or 1.0) for c, n in T.REASON_CELLS}
    isq = {key: (v - t["w"]) ** 2 for key, v in ideal.items()}

    def curve(sq, idx):  # the no-description error curve on G (nan at 0 and where a cell was not run)
        return np.array([np.nanmean(sq[none, n][idx]) if n else np.nan for n in G])

    def at(sq, c, n, idx):  # a description's error at n, on G
        return np.array([np.nanmean(sq[c, n][idx]) if j == n else np.nan for j in G])

    rows, contrasts = [], []
    for name, kind, path in RUNS:
        if not path.exists():
            continue
        pred, tokens = load(kind, path)
        if kind == "dir" and np.isnan(tokens).mean() > 0.01:  # a pass still running
            print(f"{name}: {np.isfinite(tokens).mean():.0%} of replies in, skipped")
            continue
        sq = {key: (pred[k] - t["w"]) ** 2 for key, k in cell.items()}
        for c in conds:
            row = dict(run=name, condition=T.label(c).replace("noise 60, ", ""))
            for n in (1, 2, 4):
                k, idx = cell[c, n], slice(None)
                ok = np.isfinite(pred[k])
                m, ybar = T.stated(t, c).astype(float), t["y"][:, :n].mean(1)
                row.update({f"number{n}": ok.mean(), f"thinking{n}": np.mean(tokens[k][ok] > 0), f"tokens{n}": np.nanmean(tokens[k]),
                            f"weight{n}": A.fit(pred[k][ok], m[ok], ybar[ok])[0], f"ideal_weight{n}": A.fit(ideal[c, n][ok], m[ok], ybar[ok])[0],
                            f"error{n}": np.sqrt(np.nanmean(sq[c, n])), f"ideal_error{n}": np.sqrt(isq[c, n][ok].mean())})  # Bayes on the same tasks
                row[f"ess{n}"], row[f"censored{n}"] = A.worth(curve(sq, idx), at(sq, c, n, idx), n, G)
                row[f"ideal_ess{n}"] = A.worth(curve(isq, idx), at(isq, c, n, idx), n, G)[0]
                # in reference units: the demonstrations the Bayes-optimal predictor without a description needs to match the model with it
                row[f"ref_ess{n}"] = A.worth(curve(isq, idx), at(sq, c, n, idx), n, G)[0]
            for key in ("ess", "ideal_ess", "ref_ess"):
                row[key] = float(np.median([row[f"{key}{n}"] for n in (1, 2, 4)]))
            row["censored"] = any(row[f"censored{n}"] for n in (1, 2, 4))  # one bound makes the median a bound, as in analyze.py
            for key, ref in (("ess", sq), ("ref_ess", isq)):
                draws = [float(np.median([A.worth(curve(ref, i), at(sq, c, n, i), n, G)[0] for n in (1, 2, 4)])) for i in boots]
                row[f"{key}_lo"], row[f"{key}_hi"] = np.percentile(draws, 2.5), np.percentile(draws, 97.5)
            rows.append(row)
        for acc in (10, 60):  # the specificity control's primary contrast, as in control.py: stated sd 60 -> 10, value and numbers fixed
            lo = next(c for c in conds if c[1] == 60 and c[2] and c[3] is None and (c[4] if len(c) > 4 else 60) == acc)
            hi = next(c for c in conds if c[1] == 10 and c[2] and c[3] is None and (c[4] if len(c) > 4 else 10) == acc)
            dm = np.stack([pred[cell[hi, n]] - pred[cell[lo, n]] for n in (1, 2, 4)], 1)
            di = np.stack([ideal[hi, n] - ideal[lo, n] for n in (1, 2, 4)], 1)

            def slope(i):
                x, y = di[i].ravel(), dm[i].ravel()
                keep = np.isfinite(y)
                return np.polyfit(x[keep], y[keep], 1)[0] if keep.sum() > 2 else np.nan  # a run still in progress

            draws = [slope(i) for i in boots]
            contrasts.append(dict(model=name, actual=acc, share=slope(slice(None)), lo=np.percentile(draws, 2.5), hi=np.percentile(draws, 97.5)))
        print(f"{name}: number {np.isfinite(pred).mean():.3f}, tokens {np.nanmean(tokens):.0f}; no-description error " +
              " ".join(f"{n}:{np.sqrt(np.nanmean(sq[none, n])):.1f}/{np.sqrt(isq[none, n].mean()):.1f}" for n in G[1:]))
    with open(here / "reasoning.csv", "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    with open(here / "reason_contrast.csv", "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(contrasts[0]))
        out.writeheader()
        out.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in contrasts)
    table(rows)
    for r in rows:
        print(f"{r['run']:18s} {r['condition']:40s} num {r['number1']:.2f} think {r['thinking1']:.2f}  w1 {r['weight1']:.2f} ({r['ideal_weight1']:.2f}) "
              f"w4 {r['weight4']:.2f} ({r['ideal_weight4']:.2f})  ESS {r['ess']:5.1f} [{r['ess_lo']:.1f}, {r['ess_hi']:.1f}]  ref {r['ref_ess']:5.1f} [{r['ref_ess_lo']:.1f}, {r['ref_ess_hi']:.1f}]  Bayes {r['ideal_ess']:.1f}")
    return rows


def table(rows):
    """reason_table.tex: one row per run, one column per description (and the control's value accurate to sd 10
    stated with sd 30): the ESS from the run's own no-description curve and, in parentheses, off the Bayes-optimal
    curve (median over n = 1, 2, 4); first row, the Bayes-optimal values."""
    order = ["GPT-6 Sol, none", "GPT-6 Sol, low", "Opus 4.6, no thinking", "Opus 4.6, high", "Gemma 4 31B, reasoning",
             "Qwen3.5 9B, reasoning", "gpt-oss-20b, low", "Haiku 5.5, medium", "Haiku 5.5, low", "Olmo 3 7B Think"]
    have = {r["run"] for r in rows}
    runs = [r for r in order if r in have]
    conds = ["relevant sd 60", "relevant sd 30", "relevant sd 10", "irrelevant sd 10", "relevant sd 10, 10% may be wrong",
             "relevant sd 10, 50% may be wrong", "relevant sd 30, value of sd 10"]
    names = ["$c = 1$", "$c = 4$", "$c = 36$", "irrelevant", "10\\% wrong", "50\\% wrong", "$c = 4$, drawn $36$"]
    shown = {"GPT-6 Sol, none": "GPT-6 Sol, without reasoning", "GPT-6 Sol, low": "GPT-6 Sol, with reasoning (low)",
             "Opus 4.6, no thinking": "Opus 4.6, without reasoning", "Opus 4.6, high": "Opus 4.6, with reasoning (high)",
             "Gemma 4 31B, reasoning": "Gemma 4 31B, with reasoning", "Qwen3.5 9B, reasoning": "Qwen3.5 9B, with reasoning",
             "gpt-oss-20b, low": "gpt-oss-20b, with reasoning (low)", "Haiku 5.5, medium": "Haiku 5.5, with reasoning (medium)",
             "Haiku 5.5, low": "Haiku 5.5, with reasoning (low)", "Olmo 3 7B Think": "Olmo 3 7B Think, with reasoning"}
    get = lambda run, cond: next(r for r in rows if r["run"] == run and r["condition"] == cond)
    f = lambda v, c=False: A.fmt_worth(v, c, 4)
    f1 = lambda v, c=False: f(v, c) if c else f"{v:.1f}"  # the Bayes-optimal row with one decimal, as the text quotes it
    out = ["\\begin{tabular}{l" + "c" * len(conds) + "}", "\\toprule", "Run & " + " & ".join(names) + " \\\\", "\\midrule",
           "Reference & " + " & ".join(f1(get(runs[0], c)["ideal_ess"], c.startswith("irrelevant")) for c in conds) + " \\\\"]
    for run in runs:
        cells = []
        for c in conds:
            r = get(run, c)
            own, ref = f(r["ess"], r["censored"]), f(r["ref_ess"], r["censored"])
            cells.append(own if own == ref else f"{own} ({ref})")
        out.append(shown.get(run, run) + " & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (here / "reason_table.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
