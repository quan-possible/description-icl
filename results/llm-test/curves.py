"""The error curves the language-model ESS is read from, and how much the reading of those curves matters.

    cd results/llm-test && python3 curves.py

writes llm_curves.pdf (the appendix figure: each model's error against the demonstrations shown, without a
description and with each relevant description) and llm_table_sensitivity.tex (the headline ESS, the median over
n = 1, 2, 4, under four readings of the no-description curve). Noise 60, the six reported models.
"""

import pathlib

import numpy as np

import analyze as A
import llm_test as T

here = pathlib.Path(__file__).parent
FILES = ["full_gptoss20", "full_phi4", "full_olmo32", "written_qwen9", "written_gemma12", "written_gemma31"]
NONE = (T.MAIN, None, True, None)
RELEVANT = [(T.MAIN, sd, True, None) for sd in T.REGIMES[T.MAIN]]  # c = 1, 4, 36
CS = [1, 4, 36]
NS = [n for n in T.NS if n >= 1]


def mse_curves(path):
    """{cond: mean squared error at each n in NS} for the model in path and for the Bayes-optimal predictor on the
    same tasks. A reply without a number is left out, as in the ESS of analyze.py."""
    d = np.load(here / f"{path}.npz")
    T.S = d["mean"].shape[1]
    t = T.make_tasks(T.MAIN)
    model, ideal = {}, {}
    for cond in [NONE] + RELEVANT:
        ci = T.CONDITIONS.index(cond)
        model[cond] = np.array([np.nanmean((d["mean"][ci, :, T.NS.index(n)] - t["w"]) ** 2) for n in NS])
        ideal[cond] = np.array([np.mean((T.ideal_mean(t, cond, n) - t["w"]) ** 2) for n in NS])
    return A.model_label(str(d["repo"])), model, ideal


def read_off(curve, target, n, log=True, monotone=False):
    """ESS_n read from a no-description curve known at NS, interpolated linearly in log n or in n, at the smallest
    crossing of the curve as observed or after a non-increasing fit. (value, censored) as in analyze.worth."""
    grid = np.log(NS) if log else np.array(NS, float)
    curve = A.decreasing(curve) if monotone else np.asarray(curve)
    below = np.nonzero(curve <= target)[0]
    if below.size == 0:
        return float("inf"), False
    k = int(below[0])
    if k == 0:
        return float(NS[0] - n), True
    x = grid[k - 1] + (grid[k] - grid[k - 1]) * (curve[k - 1] - target) / (curve[k - 1] - curve[k])
    return (float(np.exp(x)) if log else float(x)) - n, False


def read_fit(curve, target, n):
    """ESS_n read from the curve A / (n + kappa) + B fitted to the no-description errors (relative least squares,
    kappa on a grid), the form of a Bayesian learner's error with a noise floor. Censored as in read_off."""
    best = None
    for kappa in np.arange(0, 100.01, 0.05):
        X = np.column_stack([1 / (np.array(NS) + kappa), np.ones(len(NS))]) / curve[:, None]
        (a, b), *_ = np.linalg.lstsq(X, np.ones(len(NS)), rcond=None)
        if b < 0:  # no floor (the Bayes-optimal predictor): fit A alone
            a, b = float(np.linalg.lstsq(X[:, :1], np.ones(len(NS)), rcond=None)[0][0]), 0.0
        sse = float(((X @ [a, b]) - 1) ** 2 @ np.ones(len(NS)))
        if a > 0 and b >= 0 and (best is None or sse < best[0]):
            best = (sse, a, b, kappa)
    _, a, b, kappa = best
    if target <= b:
        return float("inf"), False
    t = a / (target - b) - kappa
    return (float(NS[0] - n), True) if t < NS[0] else (t - n, False)


READINGS = {  # name: ESS_n from (no-description curve, error with the description at n, n)
    "paper": lambda cur, tgt, n: read_off(cur, tgt, n),
    "linear": lambda cur, tgt, n: read_off(cur, tgt, n, log=False),
    "monotone": lambda cur, tgt, n: read_off(cur, tgt, n, monotone=True),
    "fit": read_fit,
}


def headline(curves, reading):
    """Median ESS over n = 1, 2, 4 for each relevant description; a bound if any term is one."""
    out = []
    for cond in RELEVANT:
        pts = [READINGS[reading](curves[NONE], curves[cond][NS.index(n)], n) for n in A.MEDIAN_NS]
        out.append((float(np.median([v for v, _ in pts])), any(c for _, c in pts)))
    return out


def table(results):
    names = list(READINGS)
    lines = [r"\begin{tabular}{l" + "rrrr" * len(CS) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{$c = {c}$}}" for c in CS) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(CS))),
             "Predictor & " + " & ".join(["log $n$", "linear", "monotone", "fit"] * len(CS)) + r" \\", r"\midrule"]
    label0, _, ideal = results[0]
    rows = [("Bayes-optimal", ideal)] + [(label, model) for label, model, _ in results]
    for label, curves in rows:
        by = {r: headline(curves, r) for r in names}
        cells = [A.fmt_worth(*by[r][i], max(A.MEDIAN_NS)) for i in range(len(CS)) for r in names]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
        if label == "Bayes-optimal":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (here / "llm_table_sensitivity.tex").write_text("\n".join(lines) + "\n")


def figure(results):
    """2 x 4 panels: the Bayes-optimal predictor, then each model; root-mean-square error against n on log axes.
    Blue, no description; green, darker for a more specific relevant description. The ESS is the horizontal
    distance from a green point to the blue curve."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex, to_rgb
    from matplotlib.lines import Line2D

    import paper_style as ps

    ps.use_style()
    fig, axes = plt.subplots(2, 4, figsize=(ps.WIDE, 3.6), layout="constrained", sharex=True, sharey=True)
    axes = axes.ravel()
    mix = lambda f: to_hex(f * np.array(to_rgb(ps.GREEN.edge)) + (1 - f) * np.array(to_rgb(ps.GREEN.fill)))
    greens = [mix(0.45), mix(0.8), to_hex(0.7 * np.array(to_rgb(ps.GREEN.edge)))]  # c = 1, 4, 36, light to dark
    panels = [("Bayes-optimal", results[0][2])] + [(label.replace(" (written)", ", written"), model) for label, model, _ in results]
    for ax, (label, curves) in zip(axes, panels):
        ax.plot(NS, np.sqrt(curves[NONE]), color=ps.BLUE.edge, lw=1.8, marker="o", ms=3, mfc=ps.BLUE.fill, mec=ps.BLUE.edge)
        for cond, col in zip(RELEVANT, greens):
            ax.plot(NS, np.sqrt(curves[cond]), color=col, lw=1.4, marker="s", ms=2.6, mfc=col, mec=col)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS]); ax.minorticks_off()
        ax.set_ylim(5, 80); ax.set_yticks([10, 30, 60]); ax.set_yticklabels(["10", "30", "60"])
        ax.set_title(label, fontsize=8, color=ps.TEXT, loc="left")
        ps.style_axes(ax, grid_axis="y")
    for ax in axes[4:]:
        ax.set_xlabel("demonstrations shown, $n$")
    for ax in axes[::4]:
        ax.set_ylabel("error (RMS)")
    axes[-1].axis("off")
    handles = [Line2D([], [], color=ps.BLUE.edge, lw=1.8, marker="o", ms=3, mfc=ps.BLUE.fill, label="no description")]
    handles += [Line2D([], [], color=col, lw=1.4, marker="s", ms=2.6, label=f"relevant, $c = {c}$") for c, col in zip(CS, greens)]
    axes[-1].legend(handles=handles, loc="center", frameon=False)
    ps.save(fig, str(here / "llm_curves"))


if __name__ == "__main__":
    results = [mse_curves(f) for f in FILES]
    for label, model, ideal in results:
        print(label, {r: headline(model, r) for r in READINGS})
    print("Bayes-optimal", {r: headline(results[0][2], r) for r in READINGS})
    table(results)
    figure(results)
