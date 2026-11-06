"""objective_analysis.py: the objective ablation (llm_test.py --objective --ablation, run 2026-10-21) against the same
model's prompt without the sentence. The variant adds " Your answer is scored by its squared difference from that
number." after "Predict the next number." on the no-description and the three relevant conditions (sigma = 60, tau =
60, 30, 10; c = 1, 4, 36) at n = 1, 2, 4, 2,000 tasks.

    cd results/llm-test && python3 objective_analysis.py

Per model, prompt and c: the share of the reply that is a number, the weight on the stated value at n = 1, 2, 4
(analyze.fit), and the ESS of Definition 1, median over n = 1, 2, 4 (analyze.worth), with 95% bootstrap intervals over
tasks drawn as analyze.main draws them. The objective runs have the no-description curve at n = 1, 2, 4 only, so the
main run's ESS is read twice: off its curve cut to n = 1, 2, 4 (grid "124", the like-for-like comparison) and off its
whole curve (grid "full", the paper's number). The objective rows also carry the paired bootstrap difference,
objective minus main, on grid 124. Writes objective_ess.csv and objective_table.tex.
"""

import csv
import pathlib

import numpy as np

import analyze as A
import llm_test as T

here = pathlib.Path(__file__).parent
DIG = here / "../../tmp/digits_unpack"
# model: (objective run, [(readout of the main run, main run)]); the first main run is the comparison
MODELS = {
    "gpt-oss-20b": ("objective_gptoss20", [("distribution", here / "full_gptoss20.npz")]),
    "Phi-4": ("objective_phi4", [("distribution", here / "full_phi4.npz")]),
    "OLMo 2 32B": ("objective_olmo32", [("distribution", here / "full_olmo32.npz")]),
    "Qwen3.5 9B": ("objective_qwen9", [("written", here / "written_qwen9.npz")]),  # its main digit run is missing
    "Gemma 4 12B": ("objective_gemma12", [("digits", DIG / "digits_gemma12.npz"), ("written", here / "written_gemma12.npz")]),
    "Gemma 4 31B": ("objective_gemma31", [("digits", DIG / "digits_gemma31.npz"), ("written", here / "written_gemma31.npz")]),
}
NONE, SDS = 0, T.REGIMES[T.MAIN]
CI = {sd: T.CONDITIONS.index((T.MAIN, sd, True, None)) for sd in SDS}
NI = [T.NS.index(n) for n in A.MEDIAN_NS]
T.S = 2000
TASKS = T.make_tasks(T.MAIN)
BOOTS = np.random.default_rng(0).integers(0, T.S, (A.BOOT, T.S))  # as analyze.main


def ideal_sq(ci):
    return np.stack([(T.ideal_mean(TASKS, T.CONDITIONS[ci], n) - TASKS["w"]) ** 2 for n in T.NS], 1)


def bound(r):
    """The median over n as printed: an infinite median becomes the lower bound median(value or 4 - n)."""
    if np.isfinite(r["ess"]):
        return r["ess"]
    return float(np.median([v if np.isfinite(v) else max(A.MEDIAN_NS) - n for (v, _), n in zip(r["pts"], A.MEDIAN_NS)]))


def measure(path, grid):
    """{sd: dict} for one run; grid 124 masks the no-description curve to n = 1, 2, 4."""
    d = np.load(path)
    assert d["mean"].shape[1] == T.S
    sq = lambda ci: (d["mean"][ci].astype(float) - TASKS["w"][:, None]) ** 2
    s0 = sq(NONE)
    i0 = ideal_sq(NONE)
    if grid == "124":
        cut = [k for k in range(len(T.NS)) if k not in NI]
        s0[:, cut] = np.nan
        i0[:, cut] = np.nan
    ess = lambda a, b: [A.worth(np.nanmean(a, 0), np.nanmean(b, 0), n) for n in A.MEDIAN_NS]
    out = {}
    for sd, ci in CI.items():
        s1, i1 = sq(ci), ideal_sq(ci)
        pts, ipts = ess(s0, s1), ess(i0, i1)
        draws = np.array([np.median([v for v, _ in ess(s0[i], s1[i])]) for i in BOOTS])
        m = T.stated(TASKS, T.CONDITIONS[ci]).astype(float)
        w, wb = {}, {}
        for n, k in zip(A.MEDIAN_NS, NI):
            pred, ybar = d["mean"][ci, :, k].astype(float), TASKS["y"][:, :n].mean(1)
            keep = ~np.isnan(pred)
            w[n] = A.fit(pred[keep], m[keep], ybar[keep])[0]
            wb[n] = np.array([A.fit(pred[i][keep[i]], m[i][keep[i]], ybar[i][keep[i]])[0] for i in BOOTS])
        out[sd] = dict(ess=float(np.median([v for v, _ in pts])), censored=any(c for _, c in pts), pts=pts, draws=draws,
                       ideal_ess=float(np.median([v for v, _ in ipts])), ideal_bound=bound(dict(ess=np.median([v for v, _ in ipts]), pts=ipts)), w=w, wb=wb,
                       p_number=float(np.mean(d["mass"][[NONE, ci]][:, :, NI])),
                       none_rmse={n: float(np.sqrt(np.nanmean(s0[:, k]))) for n, k in zip(A.MEDIAN_NS, NI)})
    return str(d["repo"]), out


def pct(x):
    x = np.asarray(x, float)
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if np.isfinite(x).all() else (np.nan, np.nan)


if __name__ == "__main__":
    rows, table = [], {}
    for model, (obj, mains) in MODELS.items():
        _, o = measure(DIG / f"{obj}.npz", "124")
        runs = [("objective", "digits" if "gemma" in obj or "qwen" in obj else "distribution", "124", o)]
        for k, (readout, path) in enumerate(mains):
            for grid in ("124", "full"):
                runs.append(("main", readout, grid, measure(path, grid)[1]))
        base = runs[1][3]  # the comparison: first main run, grid 124
        table[model] = (runs[1], runs[0])
        for prompt, readout, grid, res in runs:
            for sd in SDS:
                r = res[sd]
                row = dict(model=model, prompt=prompt, readout=readout, grid=grid, c=round((T.MAIN / sd) ** 2), p_number=round(r["p_number"], 3),
                           ess=r["ess"], ess_bound=bound(r), ess_censored=r["censored"], ess_lo=pct(r["draws"])[0], ess_hi=pct(r["draws"])[1],
                           ideal_ess=r["ideal_ess"], ideal_ess_bound=r["ideal_bound"], **{f"none_rmse_n{n}": round(v, 2) for n, v in r["none_rmse"].items()})
                for n in A.MEDIAN_NS:
                    row[f"weight_n{n}"], (row[f"weight_n{n}_lo"], row[f"weight_n{n}_hi"]) = r["w"][n], pct(r["wb"][n])
                if prompt == "objective":
                    b = base[sd]
                    row["d_ess_lo"], row["d_ess_hi"] = pct(r["draws"] - b["draws"])
                    for n in A.MEDIAN_NS:
                        row[f"d_weight_n{n}"] = r["w"][n] - b["w"][n]
                        row[f"d_weight_n{n}_lo"], row[f"d_weight_n{n}_hi"] = pct(r["wb"][n] - b["wb"][n])
                rows.append(row)
        print(model, "done", flush=True)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(here / "objective_ess.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    # objective_table.tex: ESS (median over n = 1, 2, 4, off the curve at n = 1, 2, 4) and weight at n = 1 and 4, with and
    # without the objective sentence; the main run is the same readout as the objective run except Qwen3.5 9B (written)
    fmt = lambda r: (A.fmt_worth(r["ess"], r["censored"], 4) if np.isfinite(r["ess"]) else rf"$> {A.fmt_worth(bound(r))}$") + (
        rf" {{\scriptsize[{A.fmt_worth(pct(r['draws'])[0])}, {A.fmt_worth(pct(r['draws'])[1])}]}}" if np.isfinite(pct(r["draws"])[0]) and not r["censored"] else "")
    ideal = measure(here / "full_gptoss20.npz", "124")[1]
    lines = [r"\begin{tabular}{llcccccccccc}", r"\toprule",
             r"& & & \multicolumn{3}{c}{ESS, median over $n = 1$, $2$, $4$} & \multicolumn{3}{c}{weight, $n = 1$} & \multicolumn{3}{c}{weight, $n = 4$} \\",
             r"\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12}",
             r"Model & Prompt & number & $c = 1$ & $c = 4$ & $c = 36$ & $c = 1$ & $c = 4$ & $c = 36$ & $c = 1$ & $c = 4$ & $c = 36$ \\", r"\midrule",
             "Bayes-optimal & & & " + " & ".join(A.fmt_worth(ideal[sd]["ideal_ess"]) if np.isfinite(ideal[sd]["ideal_ess"]) else rf"$> {A.fmt_worth(ideal[sd]['ideal_bound'])}$" for sd in SDS) + " & "
             + " & ".join(f"{c / (c + n):.2f}" for n in (1, 4) for c in (1, 4, 36)) + r" \\", r"\midrule"]
    for model, ((_, readout, _, b), (_, _, _, o)) in table.items():
        for k, (name, r) in enumerate(((f"main ({readout})" if model == "Qwen3.5 9B" else "main", b), ("loss stated", o))):
            share = np.mean([r[sd]["p_number"] for sd in SDS])
            it = lambda s: rf"\textit{{{s}}}" if share < A.MIN_NUMBER else s
            lines.append((model if k == 0 else "") + f" & {name} & {share:.2f} & " + " & ".join(it(fmt(r[sd])) for sd in SDS) + " & "
                         + " & ".join(it(f"{r[sd]['w'][n]:.2f}") for n in (1, 4) for sd in SDS) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (here / "objective_table.tex").write_text("\n".join(lines) + "\n")
