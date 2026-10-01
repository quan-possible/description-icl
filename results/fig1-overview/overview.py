"""Figure 1: a description is worth a number of examples, set by its
precision and capped by its reliability.

(a) is a schematic. (b) and (c) use d = 16, a0 = 10, sigma_y = 1, single
query (N = 1). Description regrets and ESS values are read from
results/rq2-reliability/ess_map.csv (three seeds). The examples-only regret
curve in (b) is recomputed here with seeds 100-102, as in that script.

    uv run python results/fig1-overview/overview.py
"""

import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from descriptor_icl import gaussian as g
from orx_figstyle import BASELINE, MUTED, PALETTE, TEXT, label_ends, panel_labels, save, use_style

HERE = pathlib.Path(__file__).parent
D, A0 = 16, 10.0
PS = (1.0, 0.99, 0.9, 0.7)
COLOR = {1.0: PALETTE["blue"], 0.99: PALETTE["green"], 0.9: PALETTE["orange"], 0.7: PALETTE["red"]}
MARKER = {1.0: "o", 0.99: "s", 0.9: "^", 0.7: "D"}


def ess_rows():
    with (HERE.parent / "rq2-reliability" / "ess_map.csv").open() as f:
        return [
            {k: float(v) for k, v in row.items()}
            for row in csv.DictReader(f)
            if int(row["d"]) == D and float(row["a0"]) == A0 and int(row["N"]) == 1
        ]


def examples_regret(K=140, S=20_000, seeds=(100, 101, 102)):
    """Single-query regret after n examples under the base prior, n = 0..K-1."""
    G = np.mean(
        [g.G_path(g.info_increments(g.gaussian_designs(D, K, S, s), A0)[0])[0] for s in seeds],
        axis=0,
    )
    return g.regret_examples(G, 1)


def token(ax, x, y, text, w=0.9, face="white", edge=BASELINE):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.42), w, 0.84, boxstyle="round,pad=0,rounding_size=0.15",
                                facecolor=face, edgecolor=edge, linewidth=0.6))
    ax.text(x, y, text, ha="center", va="center", fontsize=7)


def schematic(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    accent = PALETTE["orange"]
    top, bot = 3.0, 1.0

    ax.text(0.0, top, "Description only", ha="left", va="center", fontsize=7, color="#404040")
    ax.text(0.0, bot, "Examples only", ha="left", va="center", fontsize=7, color="#404040")
    token(ax, 2.35, top, "description $m$", w=1.5, face="#FBEFD9", edge=accent)
    token(ax, 3.55, top, "$x$", w=0.6)
    for i, t in enumerate(("$x_1, y_1$", "$x_2, y_2$", "$\\cdots$", "$x_n, y_n$")):
        token(ax, 1.95 + 0.82 * i, bot, t, w=0.74, face="white" if t != "$\\cdots$" else "none",
              edge=BASELINE if t != "$\\cdots$" else "none")
    token(ax, 5.15, bot, "$x$", w=0.6)
    for y in (top, bot):
        x0 = 3.85 if y == top else 5.45
        ax.annotate("", xy=(5.95, y), xytext=(x0 + 0.05, y),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color=BASELINE, mutation_scale=6))
        ax.text(6.05, y, "predict $y$", ha="left", va="center", fontsize=7)
    ax.text(6.05, 2.0, "equal regret\nwhen $n$ = ESS", ha="left", va="center", fontsize=7,
            fontweight="bold", color="#202020")
    ax.plot([6.0, 6.0], [1.45, 2.55], color="#202020", lw=0.6)

    # The description as a prior on w: a robust mixture. Shapes are illustrative.
    inset = ax.inset_axes([0.735, 0.04, 0.265, 0.92])
    inset.axis("off")
    w = np.linspace(-3, 3, 600)
    p, r, m = 0.75, 0.04, 1.3
    base = np.exp(-w**2 / 2) / np.sqrt(2 * np.pi)
    desc = np.exp(-(w - m) ** 2 / (2 * r)) / np.sqrt(2 * np.pi * r)
    inset.fill_between(w, (1 - p) * base, color=MUTED, lw=0)
    inset.fill_between(w, p * desc, color=accent, alpha=0.35, lw=0)
    inset.plot(w, p * desc + (1 - p) * base, color=accent, lw=1.0)
    inset.axhline(0, color="#404040", lw=0.6)
    inset.set_xlim(-3, 3.6)
    inset.set_ylim(-0.25, 1.75)
    inset.text(3.6, -0.08, "$w$", ha="right", va="top", fontsize=7)
    inset.annotate("reliability $p$", xy=(m + 0.12, 1.2), xytext=(1.8, 1.55),
                   fontsize=7, ha="left", va="center",
                   arrowprops=dict(arrowstyle="-", lw=0.5, color="#404040"))
    inset.annotate("precision $1/r$", xy=(m + 0.22, 0.45), xytext=(1.8, 0.75),
                   fontsize=7, ha="left", va="center",
                   arrowprops=dict(arrowstyle="-", lw=0.5, color="#404040"))
    inset.text(-1.5, 0.12, "wrong: $1-p$", ha="center", va="bottom", fontsize=7, color="#606060")
    return inset


def main():
    use_style()
    rows = ess_rows()
    R = examples_regret()

    fig = plt.figure(figsize=(TEXT, 3.35), layout="constrained")
    axd = fig.subplot_mosaic([["a", "a"], ["b", "c"]], height_ratios=[0.75, 1.3])
    schematic(axd["a"])

    # (b) the definition, on the most precise description in the grid
    ax = axd["b"]
    n = np.arange(len(R))
    ax.plot(n, R, color="#202020", lw=1.2)
    ax.text(24, 1.3, "examples only", fontsize=7, ha="left", va="bottom")
    for p in (1.0, 0.9):
        row = next(x for x in rows if x["r"] == 0.001 and x["p"] == p)
        crossing = g.first_crossing(R, row["regret"])[1]
        assert abs(crossing - row["ess"]) < 0.03 * row["ess"], (p, crossing, row["ess"])
        print(f"p={p:g}: curve crossing {crossing:.2f}, ess_map.csv {row['ess']:.2f}")
        c = COLOR[p]
        ax.plot([0, row["ess"]], [row["regret"]] * 2, color=c, lw=1.0, ls="--")
        ax.plot([row["ess"]] * 2, [0.04, row["regret"]], color=c, lw=0.8, ls=":")
        ax.plot(row["ess"], row["regret"], marker=MARKER[p], color=c, ms=4)
        left = p == 1.0
        ax.text(row["ess"] + (-2 if left else 2), 0.046, f"ESS = {row['ess']:.0f}", color=c,
                fontsize=7, ha="right" if left else "left", va="bottom")
        ax.text(2, row["regret"] * (1.1 if left else 0.88), f"description, $p$ = {p:g}" if left else f"$p$ = {p:g}", color=c,
                fontsize=7, ha="left", va="bottom" if left else "top",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
    ax.set_yscale("log")
    ax.set_xlim(0, 130)
    ax.set_ylim(0.04, 4)
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2], ["0.05", "0.1", "0.2", "0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_xlabel("In-context examples $n$")
    ax.set_ylabel("One-step regret (nats)")

    # (c) ESS against precision, one line per reliability
    ax = axd["c"]
    lines = []
    for p in PS:
        pts = sorted((1 / x["r"], x["ess"]) for x in rows if x["p"] == p)
        xs, ys = zip(*pts)
        (line,) = ax.plot(xs, ys, color=COLOR[p], marker=MARKER[p], ms=3.5)
        lines.append(line)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 20, 100, 1000], ["2", "5", "20", "100", "1000"])
    ax.minorticks_off()
    ax.set_ylim(0, 125)
    ax.set_xlabel("Description precision $1/r$")
    ax.set_ylabel("ESS (examples)")
    label_ends(ax, lines, [f"$p$ = {p:g}" for p in PS], pad=4)

    panel_labels([axd["a"], axd["b"], axd["c"]])
    save(fig, str(HERE / "overview"))


if __name__ == "__main__":
    main()
