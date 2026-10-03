"""Results of the language-model test from the arrays score_model.py saved.

For each model, wording (main: "Mean: m ± tol"; sentence; interval) and stated
tolerance:
- the implied strength of the line: the weight of the stated value in the mean
  of the model's predictive distribution, in numbers (n a_n / b_n over n = 5..10),
  for valid and invalid lines, and its change with the stated tolerance (paired
  bootstrap over tasks);
- the content ESS of a valid line: Definition 1 with the invalid line of the same
  wording and tolerance as the reference curve, over the number tokens and with
  the full next-token loss, in the window n = 5..10 and in later windows;
- the ESS against the prompt with no line (Definition 1 as stated);
- the same two ESS under squared error of the predictive mean;
- the probability the model puts on exactly the stated value.
Also the regret curves and the ideal learner's regret on the same tasks.

Writes summary.csv, curves.csv, llm_table.tex, llm_table_full.tex and llm_test.pdf.

    cd results/llm-test && python3 analyze.py base7.npz instruct7.npz base13.npz instruct13.npz base32.npz instruct32.npz
"""

import csv
import pathlib
import sys

import numpy as np

import llm_test as T

here = pathlib.Path(__file__).parent
BOOT = 1000
WORDINGS = ("main", "sentence", "interval")
WINDOWS = {"5-10": range(5, 11), "11-20": range(11, 21), "21-40": range(21, 41), "41-80": range(41, 81)}


def cond(kind, tol, wording):
    return f"{kind}{tol}" + ("" if wording == "main" else f"_{wording}")


def load(path):
    z = np.load(path, allow_pickle=True)
    names = list(z["conditions"])
    d = {k: {c: z[k][i] for i, c in enumerate(names)} for k in z.files if z[k].ndim == 3}
    repo, tasks = str(z["repo"]), T.make_tasks(int(z["tasks"]), int(z["seed"]))
    r = np.corrcoef(d["mean"]["none"][:, -1], tasks["y"][:, :-1].mean(1))[0, 1]
    assert r > 0.9, f"{path}: the regenerated tasks do not match the scored ones (correlation {r:.2f})"
    return repo, d, tasks


def label(repo):
    """'7B', '7B instruct', ... from the repository name."""
    size = next(p for p in repo.split("-") if p.endswith("B") and p[:-1].isdigit())
    return size + (" instruct" if repo.endswith("Instruct") else "")


def loss(d, tasks, c, reading):
    """Per-task loss curves (S, K): regret over the number tokens ('numbers'), regret with the full
    next-token loss ('all'), or squared error of the predictive mean about the hidden mean ('squared')."""
    if reading == "squared":
        return (d["mean"][c] - tasks["w"][:, None]) ** 2
    return d["regret"][c] - (np.log(d["mass"][c]) if reading == "all" else 0)


def boot(stat, S, rng):
    """Point value and 95% bootstrap interval over tasks of stat(index array)."""
    draws = np.array([stat(rng.integers(0, S, S)) for _ in range(BOOT)])
    ok = draws[np.isfinite(draws)]
    lo, hi = np.percentile(ok, [2.5, 97.5]) if ok.size > BOOT / 2 else (np.nan, np.nan)
    return stat(np.arange(S)), lo, hi


def strength(d, tasks, c, idx, window=T.WINDOW, without_copying=False):
    """Implied strength of the line of condition c on the tasks idx. without_copying removes the
    probability on exactly the stated number from the predictive distribution first."""
    m, mean = tasks["m"][T.CONDITIONS[c][0]][idx].astype(float), d["mean"][c][idx]
    if without_copying:
        p = d["pstated"][c][idx]
        mean = (mean - p * m[:, None]) / (1 - p)
    return T.implied_c(mean, m, tasks["y"][idx], window)


def bayes_yardsticks(tasks, block=200):
    """Yardsticks for the Bayes-optimal predictor that gives a line prior probability p of being
    valid (it hedges), per tolerance: the implied strength of a valid and of an invalid line, its
    ESS against no line, and its content ESS (number tokens, window 5..10)."""
    S, out = len(tasks["w"]), []
    mu0, pv0, L0 = T.gaussian_path(tasks["y"], T.MEAN, T.S0**2)
    none = T.gaussian_regret(tasks["w"], mu0, pv0).mean(0)
    for tol in T.TOLS:
        for p in (0.5, 0.9):
            reg = {}
            for kind in ("valid", "invalid"):
                mu1, pv1, L1 = T.gaussian_path(tasks["y"], tasks["m"][f"{kind}{tol}"], tol**2)
                pi = 1 / (1 + (1 - p) / p * np.exp(np.clip(L0 - L1, -700, 700)))
                total = np.zeros(T.K)
                for a in range(0, S, block):  # the mixture's regret needs its whole distribution
                    i = slice(a, a + block)
                    q = pi[i, :, None] * T.gaussian_pmf(mu1[i], pv1[i]) + (1 - pi[i, :, None]) * T.gaussian_pmf(mu0[i], pv0[i])
                    total += T.summarize(q, tasks["w"][i], tasks["y"][i])[0].sum(0)
                reg[kind] = total / S
            strength = {kind: T.implied_c(T.hedged_mean(tasks, f"{kind}{tol}", p), tasks["m"][f"{kind}{tol}"].astype(float), tasks["y"]) for kind in ("valid", "invalid")}
            out.append(dict(tolerance=tol, p=p, valid_strength=round(strength["valid"], 2), invalid_strength=round(strength["invalid"], 3),
                            ess_against_no_line=round(T.ess(none, reg["valid"]), 2), content_ess=round(T.ess(reg["invalid"], reg["valid"]), 2)))
    return out


def ideal_regret(tasks, c):
    """The ideal learner that takes the line as valid (closed form), mean over tasks, (K,)."""
    spec = T.CONDITIONS[c]
    mu, pv, _ = T.gaussian_path(tasks["y"], tasks["m"][spec[0]], max(spec[1], 1e-3) ** 2) if spec else T.gaussian_path(tasks["y"], T.MEAN, T.S0**2)
    return T.gaussian_regret(tasks["w"], mu, pv).mean(0)


def main(paths):
    rng = np.random.default_rng(0)
    runs = sorted((load(p) for p in paths), key=lambda r: (int(label(r[0]).split()[0][:-1]), r[0].endswith("Instruct")))
    rows, curves = [], []
    for repo, d, tasks in runs:
        name, S = label(repo), len(tasks["w"])
        print(f"\n== {name}", flush=True)
        for c in d["regret"]:
            mass, ideal = d["mass"][c].mean(0), ideal_regret(tasks, c)
            for n in range(T.K):
                curves.append(dict(model=name, condition=c, n=n, regret_numbers=round(float(d["regret"][c][:, n].mean()), 5),
                                   regret_all=round(float(loss(d, tasks, c, "all")[:, n].mean()), 5), mass=round(float(mass[n]), 4), ideal=round(float(ideal[n]), 5)))
        for c, spec in T.CONDITIONS.items():
            if not spec or c not in d["regret"]:
                continue
            key, tol, wording = spec
            valid = not key.startswith("invalid")
            row = dict(model=name, condition=c, tolerance=tol, wording=wording, valid=valid)
            for reading in ("numbers", "all", "squared"):
                none, desc = loss(d, tasks, "none", reading), loss(d, tasks, c, reading)
                e, lo, hi = boot(lambda i: T.ess(none[i].mean(0), desc[i].mean(0)), S, rng)
                row.update({f"ess_{reading}": round(e, 2), f"ess_{reading}_lo": round(lo, 2), f"ess_{reading}_hi": round(hi, 2)})
                inv = cond("invalid", tol, wording)
                if valid and tol and inv in d["regret"]:  # against the same line with another task's value
                    ref = loss(d, tasks, inv, reading)
                    e, lo, hi = boot(lambda i: T.ess(ref[i].mean(0), desc[i].mean(0)), S, rng)
                    row.update({f"content_{reading}": round(e, 2), f"content_{reading}_lo": round(lo, 2), f"content_{reading}_hi": round(hi, 2)})
                    if reading == "numbers":
                        for w, window in WINDOWS.items():
                            row[f"content_numbers_{w}"] = round(T.ess(ref.mean(0), desc.mean(0), window=window), 2)
            s, lo, hi = boot(lambda i: strength(d, tasks, c, i), S, rng)
            row.update(strength=round(s, 2), strength_lo=round(lo, 2), strength_hi=round(hi, 2))
            if valid and tol in (10, 3):  # the change in strength from the next looser tolerance, same tasks
                looser = cond("valid", {10: 30, 3: 10}[tol], wording)
                ch, lo, hi = boot(lambda i: strength(d, tasks, c, i) - strength(d, tasks, looser, i), S, rng)
                row.update(strength_change=round(ch, 2), strength_change_lo=round(lo, 2), strength_change_hi=round(hi, 2))
            for w, window in WINDOWS.items():
                row[f"strength_{w}"] = round(strength(d, tasks, c, np.arange(S), window), 2)
            if "pstated" in d:
                row["p_stated"] = round(float(d["pstated"][c][:, 5:11].mean()), 4)
                row["strength_without_copying"] = round(strength(d, tasks, c, np.arange(S), without_copying=True), 2)
            rows.append(row)
            print("   " + ", ".join(f"{k}={v}" for k, v in row.items() if k != "model"), flush=True)
    bayes = bayes_yardsticks(runs[0][2])
    print("\nBayes-optimal yardsticks:", bayes, flush=True)
    for file, data in (("summary.csv", rows), ("curves.csv", curves), ("bayes.csv", bayes)):
        with (here / file).open("w", newline="") as f:
            out = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            out.writeheader(); out.writerows(data)
    return rows, runs, bayes


def tables(rows, runs, bayes):
    """llm_table.tex (main text, tolerance 10) and llm_table_full.tex (appendix, every tolerance)."""
    by = {(r["model"], r["condition"]): r for r in rows}
    models = [label(r[0]) for r in runs]
    ideal_none = ideal_regret(runs[0][2], "none")
    num = lambda v: "---" if v is None or v != v else f"${0.0 if abs(v) < 0.05 else v:.1f}$"
    span = lambda tol, k, digits: " to ".join(f"{b[k]:.{digits}f}" for b in bayes if b["tolerance"] == tol)  # p = 0.5 to 0.9
    get = lambda m, c, k: by.get((m, c), {}).get(k)
    names = dict(main=r"$\pm$", sentence="sentence", interval="interval")
    lines = [r"\begin{tabular}{lrrrrrrrrrrrrrr}", r"\toprule",
             r"& \multicolumn{3}{c}{Strength, valid} & \multicolumn{3}{c}{Strength, invalid} & \multicolumn{3}{c}{Content ESS} & \multicolumn{3}{c}{ESS vs.\ no line} & \multicolumn{2}{c}{Regret, no line} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}\cmidrule(lr){14-15}",
             r"Model & $\pm$ & sent. & int. & $\pm$ & sent. & int. & $\pm$ & sent. & int. & $\pm$ & sent. & int. & $n = 5$ & $n = 50$ \\", r"\midrule",
             rf"Bayes-optimal, line taken as valid & \multicolumn{{3}}{{c}}{{9}} & \multicolumn{{3}}{{c}}{{---}} & \multicolumn{{3}}{{c}}{{---}} & \multicolumn{{3}}{{c}}{{8.9}} & {ideal_none[5]:.3f} & {ideal_none[50]:.3f} \\",
             rf"Bayes-optimal, hedging & \multicolumn{{3}}{{c}}{{{span(10, 'valid_strength', 1)}}} & \multicolumn{{3}}{{c}}{{{span(10, 'invalid_strength', 2)}}} & \multicolumn{{3}}{{c}}{{{span(10, 'content_ess', 1)}}} & \multicolumn{{3}}{{c}}{{{span(10, 'ess_against_no_line', 1)}}} & & \\", r"\midrule"]
    for (repo, d, _), m in zip(runs, models):
        cells = [get(m, cond("valid", 10, w), "strength") for w in WORDINGS] + [get(m, cond("invalid", 10, w), "strength") for w in WORDINGS]
        cells += [get(m, cond("valid", 10, w), "content_numbers") for w in WORDINGS] + [get(m, cond("valid", 10, w), "ess_numbers") for w in WORDINGS]
        none = d["regret"]["none"].mean(0)
        lines.append(f"{m} & " + " & ".join(num(v) for v in cells) + f" & {none[5]:.3f} & {none[50]:.3f} \\\\")
    (here / "llm_table.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")

    full = [r"\begin{tabular}{llrrrrrrrrrrrr}", r"\toprule",
            r"& & \multicolumn{3}{c}{Strength, valid} & \multicolumn{3}{c}{Strength, invalid} & \multicolumn{3}{c}{Content ESS} & \multicolumn{3}{c}{Content ESS, all} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}",
            r"Model & Wording & $\pm 30$ & $\pm 10$ & $\pm 3$ & $\pm 30$ & $\pm 10$ & $\pm 3$ & $\pm 30$ & $\pm 10$ & $\pm 3$ & $\pm 30$ & $\pm 10$ & $\pm 3$ \\", r"\midrule",
            r"Bayes-optimal & taken as valid & 1 & 9 & 100 & --- & --- & --- & --- & --- & --- & --- & --- & --- \\",
            r" & hedging, $p = 0.5$ & " + " & ".join(f"{next(b[k] for b in bayes if b['tolerance'] == t and b['p'] == 0.5):.{dg}f}" for k, dg in (("valid_strength", 1), ("invalid_strength", 2), ("content_ess", 1)) for t in T.TOLS) + r" & --- & --- & --- \\",
            r" & hedging, $p = 0.9$ & " + " & ".join(f"{next(b[k] for b in bayes if b['tolerance'] == t and b['p'] == 0.9):.{dg}f}" for k, dg in (("valid_strength", 1), ("invalid_strength", 2), ("content_ess", 1)) for t in T.TOLS) + r" & --- & --- & --- \\", r"\midrule"]
    for m in models:
        for w in WORDINGS:
            cells = [get(m, cond("valid", t, w), "strength") for t in T.TOLS] + [get(m, cond("invalid", t, w), "strength") for t in T.TOLS]
            cells += [get(m, cond("valid", t, w), "content_numbers") for t in T.TOLS] + [get(m, cond("valid", t, w), "content_all") for t in T.TOLS]
            full.append(f"{m if w == 'main' else ''} & {names[w]} & " + " & ".join(num(v) for v in cells) + r" \\")
    (here / "llm_table_full.tex").write_text("\n".join(full + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def figure(rows, runs):
    """llm_test.pdf: the implied strength of a valid line against its stated tolerance, one panel per
    wording, every model (dashed: instruction-tuned); dotted, the Bayes-optimal strength c."""
    sys.path.insert(0, str(here.parents[2] / "description-icl" / "results" / "rq2-reliability"))
    import orx_figstyle as fs

    fs.use_style()
    fig, axes = fs.figure_grid(1, 3, width=fs.TEXT, ratio=0.38)
    by = {(r["model"], r["condition"]): r for r in rows}
    sizes = sorted({label(r[0]).split()[0] for r in runs}, key=lambda s: int(s[:-1]))
    shade = dict(zip(sizes, fs.family("blue", len(sizes)) if len(sizes) > 1 else [fs.PALETTE["blue"]]))
    x = np.arange(len(T.TOLS))
    titles = dict(main="Mean: $m \\pm$ tol", sentence="sentence", interval="interval")
    for ax, w in zip(axes, WORDINGS):
        ax.plot(x, [T.SIG**2 / t**2 for t in T.TOLS], color=fs.BASELINE, lw=1, ls=":", marker="o", ms=2.5, label="Bayes-optimal")
        for repo, _, _ in runs:
            m, tuned = label(repo), repo.endswith("Instruct")
            pts = [by.get((m, cond("valid", t, w)), {}).get("strength", np.nan) for t in T.TOLS]
            ax.plot(x, pts, color=shade[m.split()[0]], lw=1.3, ls="--" if tuned else "-", marker="o", ms=2.5, mfc="white" if tuned else shade[m.split()[0]], label=m)
        ax.set_yscale("log"); ax.set_ylim(0.1, 200); ax.set_yticks([1, 10, 100]); ax.set_yticklabels(["1", "10", "100"]); ax.minorticks_off()
        ax.set_xticks(x); ax.set_xticklabels([f"$\\pm {t}$" for t in T.TOLS]); ax.set_xlim(-0.25, 2.25)
        ax.set_xlabel("stated tolerance"); ax.text(0.04, 0.95, titles[w], transform=ax.transAxes, va="top")
        ax.grid(True, axis="y", color=fs.MUTED, lw=0.5)
    for ax in axes:
        ax.set_ylabel("implied strength (numbers)")
    fs.panel_labels(axes)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=7, frameon=False, fontsize=6, handlelength=1.8, columnspacing=1.0)
    fs.save(fig, str(here / "llm_test"))


if __name__ == "__main__":
    rows, runs, bayes = main(sys.argv[1:])
    tables(rows, runs, bayes)
    figure(rows, runs)
