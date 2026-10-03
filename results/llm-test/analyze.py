"""Summary, tables and figure of the language-model test (paper Section 7) from the scored arrays.

    cd results/llm-test && python3 analyze.py full_gptoss20.npz full_phi4.npz full_minicpm2.npz full_olmo32.npz full_olmo7.npz full_olmo13.npz

writes summary.csv (every model, condition and n), medians.csv (the headline worth), llm_table.tex
(Table 2), llm_table_worth.tex and llm_table_full.tex (the appendix tables) and llm_test.pdf (Figure 4). Models that answer with a number in fewer
than three quarters of their replies appear only in the compliance column of the appendix table.
"""

import csv
import pathlib
import sys

import numpy as np

import llm_test as T

here = pathlib.Path(__file__).parent
BOOT = 500
MEDIAN_NS = (1, 2, 4)  # the headline worth is the median over these numbers of examples shown
MIN_NUMBER = 0.75
LABELS = {
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "microsoft/phi-4": "Phi-4",
    "openbmb/MiniCPM5-2B": "MiniCPM5 2B",
    "allenai/OLMo-2-0325-32B-Instruct": "OLMo 2 32B",
    "allenai/OLMo-2-1124-13B-Instruct": "OLMo 2 13B Instruct",
    "allenai/OLMo-2-1124-7B-Instruct": "OLMo 2 7B Instruct",
    "Qwen/Qwen3.5-9B": "Qwen3.5 9B",
    "google/gemma-4-12B-it": "Gemma 4 12B",
    "google/gemma-4-31B-it": "Gemma 4 31B",
}
PARAPHRASE_WORDING = ["normal, mean $m$, sd $\\tau$ (the paper's)", "about $m$, give or take $\\tau$", "Mean: $m \\pm \\tau$", "earlier estimate $m$, s.e.\\ $\\tau$"]


def model_label(repo):
    """'repo (written) (paraphrase K)' -> 'Label (written) (paraphrase K)'."""
    base, *tags = repo.split(" (")
    return LABELS.get(base, base) + "".join(" (" + t for t in tags)


def plain(models):
    """The models read from their first-token distribution: the figure and the per-n tables show only these."""
    return [m for m in models if "(" not in m]


def decreasing(y):
    """Non-increasing least-squares fit (pool adjacent violators)."""
    blocks = []
    for v in map(float, y):
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            (a, na), (b, nb) = blocks[-2], blocks[-1]
            blocks[-2:] = [[(a * na + b * nb) / (na + nb), na + nb]]
    return np.concatenate([[v] * k for v, k in blocks])


def worth(mse_none, mse_desc, n):
    """Definition 1 with squared error: the further demonstrations the no-description learner needs to match the
    error with the description at n. The no-description curve is taken from n = 1 on (at n = 0 some models
    rarely answer with a number), made non-increasing, and interpolated in log n. Returns (value, censored):
    inf when the curve never gets that low within the grid; (1 - n, True) when even one demonstration alone
    beats the description, so that the ESS is at most 1 - n."""
    ns = [k for k in T.NS if k >= 1 and np.isfinite(mse_none[T.NS.index(k)])]  # the ablation files have only n = 1, 2, 4
    grid, curve = np.log(np.array(ns, float)), decreasing([mse_none[T.NS.index(k)] for k in ns])
    target = mse_desc[T.NS.index(n)]
    if not np.isfinite(target):
        return float("nan"), False
    if target < curve[-1]:
        return float("inf"), False
    k = int(np.argmax(curve <= target))
    if k == 0:
        return float(ns[0] - n), True
    x = grid[k - 1] + (grid[k] - grid[k - 1]) * (curve[k - 1] - target) / max(curve[k - 1] - curve[k], 1e-12)
    return float(np.exp(x)) - n, False


def fit(pred, m, ybar):
    """Weights on the stated value and on the mean of the numbers shown (ybar None before any number)."""
    X = [m, np.ones(len(m))] + ([] if ybar is None else [ybar])
    coef = np.linalg.lstsq(np.column_stack(X), pred, rcond=None)[0]
    return coef[0], (np.nan if ybar is None else coef[2])


def main(paths):
    """rows: one per model, condition and n (summary.csv). medians: {(model, cond): (worth, lo, hi, ideal)}, the
    median worth over MEDIAN_NS with a 95% bootstrap interval over tasks (medians.csv)."""
    rows, medians = [], {}
    for path in paths:
        d = np.load(path)
        repo = str(d["repo"])
        T.S = d["mean"].shape[1]
        tasks = {noise: T.make_tasks(noise) for noise in T.REGIMES}
        boots = np.random.default_rng(0).integers(0, T.S, (BOOT, T.S))
        rms = lambda e: float(np.sqrt((e**2).mean()))
        conds = T.CONDITIONS[: d["mean"].shape[0]]  # older arrays lack the prior-stated condition
        sq = {ci: (d["mean"][ci].astype(float) - tasks[cond[0]]["w"][:, None]) ** 2 for ci, cond in enumerate(conds)}  # (task, n)
        ideal_sq = {ci: np.stack([T.ideal_mean(tasks[cond[0]], cond, n, cond[3] or 1.0) for n in T.NS], 1) for ci, cond in enumerate(conds)}
        ideal_sq = {ci: (v - tasks[conds[ci][0]]["w"][:, None]) ** 2 for ci, v in ideal_sq.items()}
        none = {cond[0]: ci for ci, cond in enumerate(conds) if cond[1] is None}
        for ci, cond in enumerate(conds):
            t = tasks[cond[0]]
            if cond[1] and not np.isnan(d["mean"][ci]).all():
                c0 = none[cond[0]]
                have = [n for n in MEDIAN_NS if not np.isnan(d["mean"][ci, :, T.NS.index(n)]).all()]
                if have:
                    draws = np.array([[worth(np.nanmean(sq[c0][i], 0), np.nanmean(sq[ci][i], 0), n)[0] for n in have] for i in boots])
                    med = np.median(draws, 1)
                    point = [worth(np.nanmean(sq[c0], 0), np.nanmean(sq[ci], 0), n) for n in have]
                    ideal_point = [worth(ideal_sq[c0].mean(0), ideal_sq[ci].mean(0), n) for n in have]
                    censored = lambda pts: any(c for _, c in pts)  # lowering any term can only lower the median, so one bound makes it a bound
                    medians[(model_label(repo), cond)] = (float(np.median([v for v, _ in point])), float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5)),
                                                              float(np.median([v for v, _ in ideal_point])), censored(point), censored(ideal_point))
            for ni, n in enumerate(T.NS):
                pred, ideal = d["mean"][ci, :, ni].astype(float), T.ideal_mean(t, cond, n, cond[3] or 1.0)
                if np.isnan(pred).all():
                    continue
                keep = ~np.isnan(pred)
                pred = np.where(keep, pred, np.nanmean(pred))  # a missing answer counts as the average answer
                row = dict(model=model_label(repo), noise=cond[0], sd=cond[1], valid=cond[2], p=cond[3], n=n,
                           p_number=float(d["mass"][ci, :, ni].mean()), error=rms(pred - t["w"]), ideal_error=rms(ideal - t["w"]))
                if cond[1]:
                    c0 = none[cond[0]]
                    row["ess"], row["ess_censored"] = worth(np.nanmean(sq[c0], 0), np.nanmean(sq[ci], 0), n)
                    row["ideal_ess"], row["ideal_censored"] = worth(ideal_sq[c0].mean(0), ideal_sq[ci].mean(0), n)
                    bs = np.array([worth(np.nanmean(sq[c0][i], 0), np.nanmean(sq[ci][i], 0), n)[0] for i in boots[: BOOT // 5]])
                    row["ess_lo"], row["ess_hi"] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if np.isfinite(bs).all() else (np.nan, np.nan)
                    m, ybar = T.stated(t, cond).astype(float), (t["y"][:, :n].mean(1) if n else None)
                    a, b = fit(pred, m, ybar)
                    ia, ib = fit(ideal, m, ybar)
                    row.update(weight=a, ideal_weight=ia, worth=n * a / b if n else np.nan, ideal_worth=n * ia / ib if n else np.nan,
                               weight_se=float(np.std([fit(pred[i], m[i], None if ybar is None else ybar[i])[0] for i in boots])))
                rows.append(row)
        print(path, "done", flush=True)
    with open(here / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "noise", "sd", "valid", "p", "n", "p_number", "error", "ideal_error", "ess", "ess_censored", "ideal_ess", "ideal_censored", "ess_lo", "ess_hi", "weight", "ideal_weight", "weight_se", "worth", "ideal_worth"])
        w.writeheader()
        w.writerows(rows)
    with open(here / "medians.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "noise", "sd", "valid", "p", "worth", "lo", "hi", "ideal", "censored", "ideal_censored"])
        w.writerows([m, *cond, *v] for (m, cond), v in medians.items())
    return rows, medians


def pick(rows, model=None, **k):
    out = [r for r in rows if (model is None or r["model"] == model) and all(r.get(key) == v for key, v in k.items())]
    assert len(out) == 1, (model, k, len(out))
    return out[0]


def compliant(rows):
    """Models in the order of LABELS that answer with a number at least MIN_NUMBER of the time."""
    order = []
    for r in rows:
        if r["model"] not in order and "(paraphrase" not in r["model"]:
            order.append(r["model"])
    share = lambda m, **k: np.mean([r["p_number"] for r in rows if r["model"] == m and all(r.get(a) == b for a, b in k.items())])
    # a number in at least MIN_NUMBER of all replies, and in at least half of the no-description replies over MEDIAN_NS,
    # the curve the ESS is read from (OLMo 2 32B's written readout fails the second: 5% to 48%)
    return [m for m in order if share(m) >= MIN_NUMBER and np.mean([share(m, sd=None, n=n) for n in MEDIAN_NS]) >= 0.5]


def fmt_worth(v, censored=False, n=None, top=64):
    if not np.isfinite(v) and v != float("inf"):
        return "--"
    if v == float("inf"):  # the error with the description beats the no-description error at 64 demonstrations
        return rf"$> {top - n}$"
    if censored:  # an upper bound: round up, so that the printed bound still holds
        v = np.ceil(v * 10 - 1e-9) / 10 if abs(v) < 10 else float(np.ceil(v))
    txt = f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}"
    txt = ("0.0" if txt == "-0.0" else txt).replace("-", "$-$")  # text, so that \textit can slant it; a true minus sign
    return rf"$\leq$ {txt}" if censored else txt


def label(cond):
    """A condition as the paper names it: noise, the description's Bayes-optimal ESS c, relevance, stated chance of being wrong."""
    noise, sd, valid, p = cond
    return f"$\\sigma = {noise}$, $c = {round((noise / sd) ** 2)}$, {'relevant' if valid else 'irrelevant'}" + (f", {round(100 * (1 - p))}\\% wrong" if p else "")


def tables(rows, medians):
    c36 = T.REGIMES[T.MAIN][2]
    cols = [(60, True, None), (30, True, None), (c36, True, None), (c36, False, None), (c36, True, 0.9), (c36, True, 0.5)]
    head = ["$c = 1$", "$c = 4$", "$c = 36$", "irrelevant", "stated 10\\%", "stated 50\\%"]
    wcols = [(60, True, None), (30, True, None), (c36, True, None)]
    models = compliant(rows)
    key = lambda m, c: (m, (T.MAIN, *c))
    w1 = lambda m, c, k: next(r[k] for r in rows if r["model"] == m and r["noise"] == T.MAIN and r["sd"] == c[0] and r["valid"] == c[1] and r["p"] == c[2] and r["n"] == 1)
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "c" + "c" * len(wcols) + "}", r"\toprule",
             r"& \multicolumn{6}{c}{ESS, median over $n = 1$, $2$, $4$} & & \multicolumn{3}{c}{weight, $n = 1$} \\",
             r"\cmidrule(lr){2-7} \cmidrule(lr){9-11}",
             r"& \multicolumn{3}{c}{relevant} & \multicolumn{3}{c}{$c = 36$} & & \multicolumn{3}{c}{relevant} \\",
             r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){9-11}",
             "Model & " + " & ".join(head) + " & & " + " & ".join(head[:3]) + r" \\", r"\midrule"]
    lines.append("Bayes-optimal & " + " & ".join(fmt_worth(medians[key(models[0], c)][3], medians[key(models[0], c)][5], max(MEDIAN_NS)) for c in cols) + " & & " + " & ".join(f"{w1(models[0], c, 'ideal_weight'):.2f}" for c in wcols) + r" \\")
    lines.append(r"\midrule")
    for m in models:
        cells = []
        for c in cols:
            v, lo, hi, _, cens, _ = medians[key(m, c)]
            cells.append(fmt_worth(v, cens, max(MEDIAN_NS)) + (rf" {{\scriptsize[{fmt_worth(lo)}, {fmt_worth(hi)}]}}" if np.isfinite(lo) and np.isfinite(hi) and not cens else ""))  # an infinite median: beyond the grid at n = 4 at least
        lines.append(m + " & " + " & ".join(cells) + " & & " + " & ".join(f"{w1(m, c, 'weight'):.2f}" for c in wcols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (here / "llm_table.tex").write_text("\n".join(lines) + "\n")

    # appendix: the ESS (from n = 1) and the weight on the stated value (from n = 0) at every n, the Bayes-optimal
    # predictor first, then each model; a model's cell is in italics where it answers with a number in fewer than
    # MIN_NUMBER of its replies
    models = plain(models)
    conds = [cond for cond in T.CONDITIONS if cond[1]]
    for name, key_, ideal_key in (("llm_table_worth.tex", "ess", "ideal_ess"), ("llm_table_full.tex", "weight", "ideal_weight")):
        ns = [n for n in T.NS if n >= 1] if key_ == "ess" else list(T.NS)
        show = (lambda v, cens, n: fmt_worth(v, cens, n)) if key_ == "ess" else (lambda v, cens, n: f"{v:.2f}")
        lines = [r"\begin{tabular}{llr" + "r" * len(ns) + "}", r"\toprule", r"Predictor & Description & number & " + " & ".join(f"$n = {n}$" for n in ns) + r" \\", r"\midrule"]
        for m in ["Bayes-optimal"] + models:
            src, k, ck = (models[0], ideal_key, "ideal_censored") if m == "Bayes-optimal" else (m, key_, "ess_censored")
            for i, cond in enumerate(conds):
                rs = {r["n"]: r for r in rows if r["model"] == src and (r["noise"], r["sd"], r["valid"], r["p"]) == cond}
                cells = []
                for n in ns:
                    r = rs.get(n)
                    v = show(r[k], r.get(ck), n) if r else "--"
                    cells.append(rf"\textit{{{v}}}" if r and m != "Bayes-optimal" and r["p_number"] < MIN_NUMBER else v)
                share = "" if m == "Bayes-optimal" else f"{np.mean([r['p_number'] for r in rs.values()]):.2f}"
                lines.append((m if i == 0 else "") + f" & {label(cond)} & {share} & " + " & ".join(cells) + r" \\")
            lines.append(r"\midrule")
        lines[-1] = r"\bottomrule"
        lines.append(r"\end{tabular}")
        (here / name).write_text("\n".join(lines) + "\n")
    paraphrase_table(rows, medians)


def paraphrase_table(rows, medians):
    """llm_table_paraphrase.tex: the share of replies that are a number and the weight on the stated value at n = 1 and
    n = 4 for the three relevant descriptions, under four wordings of the description sentence (written readout).
    The ESS is not shown: the ablation measures n = 1, 2, 4 only, too coarse a no-description curve to read it from."""
    sds = T.REGIMES[T.MAIN]
    bases = []
    for r in rows:
        if "(paraphrase" in r["model"]:
            b = r["model"].split(" (paraphrase")[0]
            if b not in bases:
                bases.append(b)
    if not bases:
        return
    cell = lambda m, sd, n: next(((r["weight"], r["p_number"]) for r in rows if r["model"] == m and r["noise"] == T.MAIN and r["sd"] == sd and r["valid"] and r["p"] is None and r["n"] == n), (np.nan, 0.0))
    lines = [r"\begin{tabular}{llccccccc}", r"\toprule", r"& & & \multicolumn{3}{c}{weight, $n = 1$} & \multicolumn{3}{c}{weight, $n = 4$} \\",
             r"\cmidrule(lr){4-6} \cmidrule(lr){7-9}", r"Model & Wording of the description & number & $c = 1$ & $c = 4$ & $c = 36$ & $c = 1$ & $c = 4$ & $c = 36$ \\", r"\midrule"]
    for b in bases:
        for k, wording in enumerate(PARAPHRASE_WORDING):
            m = f"{b} (paraphrase {k})"
            vals = [cell(m, sd, n) for n in (1, 4) for sd in sds]
            share = np.mean([r["p_number"] for r in rows if r["model"] == m]) if any(r["model"] == m for r in rows) else 0.0
            cells = [(rf"\textit{{{w:.2f}}}" if pn < MIN_NUMBER else f"{w:.2f}") if np.isfinite(w) else "--" for w, pn in vals]
            lines.append((b.replace(" (written)", "") if k == 0 else "") + f" & {wording} & {share:.2f} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (here / "llm_table_paraphrase.tex").write_text("\n".join(lines) + "\n")


def figure(rows, medians):
    """llm_test.pdf (Figure 4). (a) Specificity: the ESS of a relevant description for each model against its
    Bayes-optimal ESS, at c = 1, 4, 36. (b) The weight on the stated value for c = 36 minus that for c = 1, against
    the demonstrations shown, with approximate 95% intervals. (c) Stated reliability: the ESS of the relevant c = 36
    description with no stated chance of being wrong, or a stated 10% or 50%. (a) and (c) are medians over 1, 2 and
    4 demonstrations shown, with 95% bootstrap intervals, on one log scale; a value that is at most 0.3, or only an
    upper bound, is drawn as a downward triangle at the floor. Series at the same x are offset slightly."""
    import paper_style as ps

    ps.use_style()
    fig, (a, b, c) = ps.figure(3, width=ps.WIDE, height=2.15, gridspec_kw=dict(width_ratios=[1.1, 1.1, 0.8]))
    models = plain(compliant(rows))
    assert "OLMo 2 32B" in models or not any(r["model"] == "OLMo 2 32B" for r in rows), "OLMo 2 32B fails the number rule"
    tones = dict(zip(models, (ps.BLUE, ps.GREEN, ps.PINK)))  # three models, the three accents; the Bayes-optimal predictor is the dashed ink line
    colors = {m: t.edge for m, t in tones.items()}
    markers = dict(zip(models, "osD"))
    bayes = dict(zorder=4, **ps.BAYES)
    small = dict(fontsize=6.5, textcoords="offset points")
    shift = lambda k, step: step * (k - (len(models) - 1) / 2)  # offset of the k-th model at a shared x
    floor = 0.3

    def draw(ax, x, v, m, xlog):
        """v: (value, lo, hi, censored) per x on the log ESS scale; returns the line for labeling."""
        k = models.index(m)
        x = np.asarray(x, float) * 1.06 ** shift(k, 1) if xlog else np.asarray(x, float) + shift(k, 0.08)
        y = np.array([max(z[0], floor) for z in v])
        line, = ax.plot(x, y, color=colors[m], lw=1.8, label=m, marker=markers[m], ms=4, mfc=tones[m].fill, mec=colors[m], mew=0.9)
        for xi, (val, lo, hi, cens) in zip(x, v):
            if cens or val <= floor:  # a bound, or below the axis
                ax.plot(xi, floor, marker="v", ms=3.5, color=colors[m], clip_on=False, zorder=3)
            else:
                err = [[val - max(lo, floor)], [hi - val]] if np.isfinite(lo) and np.isfinite(hi) else None
                ax.errorbar(xi, val, yerr=err, color=colors[m], marker=markers[m], ms=4, mfc=tones[m].fill, mec=colors[m], mew=0.9, capsize=0, elinewidth=1, zorder=3)
        return line

    # (a) a relevant description at each specificity
    cs, sds = [1, 4, 36], T.REGIMES[T.MAIN]
    med = lambda m, k: medians[(m, (T.MAIN, *k))]
    a.plot(cs, [med(models[0], (sd, True, None))[3] for sd in sds], **bayes)
    lines = [draw(a, cs, [(z[0], z[1], z[2], z[4]) for z in (med(m, (sd, True, None)) for sd in sds)], m, xlog=True) for m in models]
    a.set_xlabel("$c$")

    # (b) how much the weight on the stated value follows the stated specificity: the weight on the most specific
    # description (c = 36) minus that on the loosest (c = 1), against the demonstrations shown
    c36 = sds[2]
    ns = [n for n in T.NS if n >= 1]
    wt = lambda m, sd, key: np.array([next(r[key] for r in rows if r["model"] == m and r["noise"] == T.MAIN and r["sd"] == sd and r["valid"] and r["p"] is None and r["n"] == n) for n in ns])
    ideal = wt(models[0], c36, "ideal_weight") - wt(models[0], sds[0], "ideal_weight")
    b.axhline(0, color=ps.SPINE, lw=0.9)
    b.plot(ns, ideal, **bayes)
    for k, m in enumerate(models):
        diff = wt(m, c36, "weight") - wt(m, sds[0], "weight")
        se = np.hypot(wt(m, c36, "weight_se"), wt(m, sds[0], "weight_se"))  # ponytail: treats the two weights as independent
        b.errorbar(np.array(ns) * 1.06 ** shift(k, 1), diff, yerr=1.96 * se, color=colors[m], marker=markers[m], ms=4, mfc=tones[m].fill, mec=colors[m], mew=0.9, lw=1.8, capsize=0, elinewidth=1)
    b.set_xlabel("demonstrations shown, $n$")
    b.set_ylabel("weight at $c = 36$ minus $c = 1$")
    b.set_ylim(-0.2, 0.8); b.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])

    # (c) the relevant c = 36 description with a stated chance of being wrong
    conds = [(c36, True, None), (c36, True, 0.9), (c36, True, 0.5)]
    xs = [0, 1, 2]
    c.plot(xs, [med(models[0], k)[3] for k in conds], **bayes)
    for m in models:
        draw(c, xs, [(z[0], z[1], z[2], z[4]) for z in (med(m, k) for k in conds)], m, xlog=False)
    c.set_xticks(xs); c.set_xticklabels(["none", "10%", "50%"])
    c.set_xlim(-0.4, 2.4)
    c.set_xlabel("stated chance wrong")

    a.set_xscale("log")
    a.set_xticks(cs); a.set_xticklabels([str(t) for t in cs]); a.minorticks_off()
    for ax in (a, c):
        ax.set_yscale("log")
        ax.set_yticks([0.3, 1, 3, 10, 30]); ax.set_yticklabels(["0.3", "1", "3", "10", "30"]); ax.set_ylim(floor, 60); ax.minorticks_off()
        ax.set_ylabel("ESS (demonstrations)")
    a.set_xlim(0.7, 36)
    b.set_xscale("log"); b.set_xticks(ns); b.set_xticklabels([str(t) for t in ns]); b.minorticks_off()
    for ax in (a, b, c):
        ps.style_axes(ax, grid_axis="y")
    ps.panel_labels([a, b, c])
    from matplotlib.lines import Line2D
    handles = lines + [Line2D([], [], label="Bayes-optimal", **ps.BAYES)]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)
    ps.save(fig, str(here / "llm_test"))


if __name__ == "__main__":
    import pickle
    cache = here / "analyze_cache.pkl"  # `python3 analyze.py replot` redraws from the last run's results
    if sys.argv[1:] == ["replot"]:
        rows, medians = pickle.loads(cache.read_bytes())
    else:
        rows, medians = main(sys.argv[1:])
        cache.write_bytes(pickle.dumps((rows, medians)))
    tables(rows, medians)
    figure(rows, medians)
