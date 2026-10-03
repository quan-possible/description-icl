"""Results of the language-model test from the arrays score_model.py saved.

For each model:
- the mean regret curves, read two ways: over the number tokens only (the saved
  regret, from the distribution renormalized over the 900 number tokens), and
  over the full vocabulary (that regret minus the log of the probability the
  model put on the number tokens);
- the ESS of each line (Definition 1 with the model's own curves, averaged over
  n = 5..10, 95% bootstrap interval over tasks), for valid and invalid lines, and
  its profile ESS_n for n = 1..30;
- the content ESS of a valid line: the same definition with the invalid line of
  the same tolerance as the reference curve in place of the no-line curve, which
  holds the form of the prompt fixed;
- the strength implied by the centre of its predictions, n a_n / b_n, with a
  bootstrap interval;
- the ideal learner's regret on the same tasks.

Writes summary.csv, curves.csv, profile.csv, llm_table.tex and llm_test.pdf here.

    python3 results/llm-test/analyze.py results/llm-test/base7.npz results/llm-test/instruct7.npz ...
"""

import csv
import pathlib
import sys

import numpy as np

import llm_test as T

here = pathlib.Path(__file__).parent
BOOT = 200
READINGS = ("numbers", "all")  # the distribution over number tokens, or the full next-token loss


def load(path):
    z = np.load(path, allow_pickle=True)
    conds = list(z["conditions"])
    d = {k: {c: z[k][i] for i, c in enumerate(conds)} for k in ("regret", "logp", "mean", "sd", "mass")}
    repo, S, seed = str(z["repo"]), int(z["tasks"]), int(z["seed"])
    tasks = T.make_tasks(S, seed)
    r = np.corrcoef(d["mean"]["none"][:, -1], tasks["y"][:, :-1].mean(1))[0, 1]
    assert r > 0.9, f"{path}: the regenerated tasks do not match the scored ones (correlation {r:.2f})"
    return repo, d, tasks


def label(repo):
    """'7B', '7B instruct', ... from the repository name."""
    size = next(p for p in repo.split("-") if p.endswith("B") and p[:-1].isdigit())
    return size + (" instruct" if repo.endswith("Instruct") else "")


def regret(d, cond, reading):
    """Per-task regret (S, K). 'all' adds the loss from probability placed off the number tokens."""
    return d["regret"][cond] - (np.log(d["mass"][cond]) if reading == "all" else 0)


def boot(stat, S, rng):
    """Point value and 95% bootstrap interval over tasks of stat(index array)."""
    draws = np.array([stat(rng.integers(0, S, S)) for _ in range(BOOT)])
    ok = draws[np.isfinite(draws)]
    lo, hi = np.percentile(ok, [2.5, 97.5]) if ok.size > BOOT / 2 else (np.nan, np.nan)
    return stat(np.arange(S)), lo, hi


def strength(d, tasks, cond, idx):
    key = T.CONDITIONS[cond][0]
    return T.implied_c(*T.weights(d["mean"][cond][idx], tasks["m"][key][idx].astype(float), tasks["y"][idx]))


def ideal_regret(tasks, cond):
    """The ideal learner that takes the line as valid (closed form), mean over tasks, (K,)."""
    spec = T.CONDITIONS[cond]
    mu, pv, _ = T.gaussian_path(tasks["y"], tasks["m"][spec[0]], max(spec[1], 1e-3) ** 2) if spec else T.gaussian_path(tasks["y"], T.MEAN, T.S0**2)
    return T.gaussian_regret(tasks["w"], mu, pv).mean(0)


def main(paths):
    rng = np.random.default_rng(0)
    runs = sorted((load(p) for p in paths), key=lambda r: (int(label(r[0]).split()[0][:-1]), r[0].endswith("Instruct")))
    rows, curves, profile = [], [], []
    for repo, d, tasks in runs:
        name, S = label(repo), len(tasks["w"])
        print(f"\n== {name}")
        for cond in d["regret"]:
            mass, ideal = d["mass"][cond].mean(0), ideal_regret(tasks, cond)
            for n in range(T.K):
                curves.append(dict(model=name, condition=cond, n=n, regret_numbers=round(float(d["regret"][cond][:, n].mean()), 5),
                                   regret_all=round(float(regret(d, cond, "all")[:, n].mean()), 5), mass=round(float(mass[n]), 4), ideal=round(float(ideal[n]), 5)))
        print("   probability on the number tokens at n = 1, 5, 10, 50: " + "; ".join(f"{c} {d['mass'][c].mean(0)[[1, 5, 10, 50]].round(2)}" for c in ("none", "valid10", "invalid10")))
        for cond, spec in T.CONDITIONS.items():
            if not spec:
                continue
            key, tol, wording = spec
            c = T.SIG**2 / tol**2 if tol else np.inf
            row = dict(model=name, condition=cond, tolerance=tol, wording=wording, valid=not key.startswith("invalid"), ideal_ess="" if key.startswith("invalid") else round(c - T.EPS, 2))
            for reading in READINGS:
                none, desc = regret(d, "none", reading), regret(d, cond, reading)
                e, lo, hi = boot(lambda i: T.ess(none[i].mean(0), desc[i].mean(0)), S, rng)
                row.update({f"ess_{reading}": round(e, 2), f"ess_{reading}_lo": round(lo, 2), f"ess_{reading}_hi": round(hi, 2)})
                for n in range(1, 31):
                    profile.append(dict(model=name, condition=cond, reading=reading, n=n, ess_n=round(T.ess(none.mean(0), desc.mean(0), window=(n,)), 3)))
                if wording == "main" and not key.startswith("invalid"):  # against the same line with another task's value
                    inv = regret(d, f"invalid{tol}", reading)
                    e, lo, hi = boot(lambda i: T.ess(inv[i].mean(0), desc[i].mean(0)), S, rng)
                    row.update({f"content_{reading}": round(e, 2), f"content_{reading}_lo": round(lo, 2), f"content_{reading}_hi": round(hi, 2)})
            if wording in ("main", "exact"):
                s, lo, hi = boot(lambda i: strength(d, tasks, cond, i), S, rng)
                row.update(strength=round(s, 2), strength_lo=round(lo, 2), strength_hi=round(hi, 2))
            rows.append(row)
            print("   " + ", ".join(f"{k}={v}" for k, v in row.items() if k != "model"))
    for file, data in (("summary.csv", rows), ("curves.csv", curves), ("profile.csv", profile)):
        with (here / file).open("w", newline="") as f:
            out = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            out.writeheader(); out.writerows(data)
    return rows, runs


def table(rows, runs):
    """llm_table.tex: per model, the ESS under both readings and the implied strength."""
    by = {(r["model"], r["condition"]): r for r in rows}
    ideal_none = ideal_regret(runs[0][2], "none")
    lines = [r"\begin{tabular}{lrrrrrrrrrrrr}", r"\toprule",
             r"& \multicolumn{4}{c}{Content ESS} & \multicolumn{2}{c}{ESS against no line} & \multicolumn{4}{c}{Implied strength} & \multicolumn{2}{c}{Regret, no line} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-11}\cmidrule(lr){12-13}",
             r"Model & $\pm 30$ & $\pm 10$ & $\pm 3$ & $\pm 10$, all & $\pm 10$ & inv. & $\pm 30$ & $\pm 10$ & $\pm 3$ & inv. & $n = 5$ & $n = 50$ \\", r"\midrule",
             rf"Ideal & 0.9 & 8.9 & 99.9 & 8.9 & 8.9 & --- & 1 & 9 & 100 & --- & {ideal_none[5]:.3f} & {ideal_none[50]:.3f} \\", r"\midrule"]
    for repo, d, _ in runs:
        m = label(repo)
        cells = [by[m, c]["content_numbers"] for c in ("valid30", "valid10", "valid3")] + [by[m, "valid10"]["content_all"]]
        cells += [by[m, c]["ess_numbers"] for c in ("valid10", "invalid10")] + [by[m, c]["strength"] for c in ("valid30", "valid10", "valid3", "invalid10")]
        none = d["regret"]["none"].mean(0)
        lines.append(f"{m} & " + " & ".join(f"${v:.1f}$" for v in cells) + f" & {none[5]:.3f} & {none[50]:.3f} \\\\")
    (here / "llm_table.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def figure(rows, runs):
    """llm_test.pdf, for the largest base model in (a) and (b): (a) regret over the number tokens
    against numbers in hand, with the ideal learner dotted; (b) the content ESS of a valid line against its stated
    tolerance, every model; (c) the weight that each base model's prediction puts on
    the stated value, valid and invalid."""
    sys.path.insert(0, str(here.parents[2] / "description-icl" / "results" / "rq2-reliability"))
    import orx_figstyle as fs

    fs.use_style()
    fig, axes = fs.figure_grid(1, 3, width=fs.TEXT, ratio=0.38)
    tones = dict(zip(T.TOLS, fs.family("blue", 3)))
    n = np.arange(T.K)
    bases = [r for r in runs if not r[0].endswith("Instruct")]
    repo, d, tasks = bases[-1]

    ax = axes[0]
    ax.plot(n[1:], d["regret"]["none"].mean(0)[1:], color=fs.BASELINE, lw=1.3, label="no line")
    ax.plot(n[1:], ideal_regret(tasks, "none")[1:], color=fs.BASELINE, lw=1, ls=":")
    for tol in T.TOLS:
        ax.plot(n[1:], d["regret"][f"valid{tol}"].mean(0)[1:], color=tones[tol], lw=1.3, label=f"$\\pm {tol}$")
        ax.plot(n[1:], ideal_regret(tasks, f"valid{tol}")[1:], color=tones[tol], lw=1, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("numbers in hand $n$"); ax.set_ylabel("regret (nats)")
    ax.legend(frameon=False, handlelength=1.2, loc="lower left", fontsize=6)

    ax = axes[1]
    by = {(r["model"], r["condition"]): r for r in rows}
    sizes = sorted({label(r[0]).split()[0] for r in runs}, key=lambda s: int(s[:-1]))
    shade = dict(zip(sizes, fs.family("blue", len(sizes)) if len(sizes) > 1 else [fs.PALETTE["blue"]]))
    x = np.arange(len(T.TOLS))
    ax.plot(x, [T.SIG**2 / t**2 - T.EPS for t in T.TOLS], color=fs.BASELINE, lw=1, ls=":", marker="o", ms=2.5, label="ideal")
    for repo, _, _ in runs:
        m, tuned = label(repo), repo.endswith("Instruct")
        ax.plot(x, [by[m, f"valid{t}"]["content_numbers"] for t in T.TOLS], color=shade[m.split()[0]], lw=1.3, ls="--" if tuned else "-",
                marker="o", ms=2.5, mfc="white" if tuned else shade[m.split()[0]], label=m)
    ax.set_yscale("log"); ax.set_ylim(0.15, 200); ax.set_yticks([1, 10, 100]); ax.set_yticklabels(["1", "10", "100"]); ax.minorticks_off()
    ax.set_xticks(x); ax.set_xticklabels([f"$\\pm {t}$" for t in T.TOLS]); ax.set_xlim(-0.25, 2.25)
    ax.set_xlabel("stated tolerance (more specific $\\rightarrow$)"); ax.set_ylabel("content ESS (numbers)")
    ax.legend(frameon=False, handlelength=1.8, fontsize=5.5, ncol=2, loc="upper left", columnspacing=1.0)
    ax.grid(True, axis="y", color=fs.MUTED, lw=0.5)

    ax = axes[2]
    for repo, d, tasks in bases:
        for kind, ls in (("valid", "-"), ("invalid", "--")):
            a, _ = T.weights(d["mean"][f"{kind}10"], tasks["m"][f"{kind}10"].astype(float), tasks["y"])
            ax.plot(n[4:], a[3:], color=shade[label(repo)], lw=1.3, ls=ls, label=f"{label(repo)}, {kind}")
    ax.plot(n[4:], 9 / (9 + n[4:]), color=fs.BASELINE, lw=1, ls=":", label="ideal, valid")
    ax.set_xscale("log"); ax.set_xlim(4, 100); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("numbers in hand $n$"); ax.set_ylabel("weight on the stated value ($\\pm 10$)")
    ax.legend(frameon=False, handlelength=1.6, fontsize=5.5)
    fs.panel_labels(axes)
    fs.save(fig, str(here / "llm_test"))


if __name__ == "__main__":
    rows, runs = main(sys.argv[1:])
    table(rows, runs)
    figure(rows, runs)
