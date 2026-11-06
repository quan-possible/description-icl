"""§6 figure: each language model's ESS without reasoning (open) and with it (filled), on the same 200 tasks, for the
c = 36 description, the specificity control (value accurate to sd 10, stated with sd 30), and the c = 36 description
with a stated 50% chance of being wrong. Dashed: the Bayes-optimal ESS. ESS from each model's own no-description curve
(Definition 1); a square, against a Bayes-optimal learner where the model's own curve never falls to its error
with the description. Right: Humanity's Last Exam (HLE) in
each mode, Artificial Analysis (accessed 4 November 2026); a dash, no score reported for the mode run here.
Data: results/llm-test/figure_data.csv (figure_data.py)."""
import csv
import math
import pathlib

import paper_style as ps

DATA = pathlib.Path(__file__).resolve().parents[3] / "results/llm-test/figure_data.csv"
# rows, ordered by HLE without reasoning; HLE (%) without, with reasoning
ROWS = [("Opus 4.6", "19.1", "–"), ("GPT-6 Sol", "18.4", "–"), ("Gemma 4 31B", "11.8", "23.6"),
        ("Qwen3.5 9B", "9.4", "14.9"), ("gpt-oss-20b", "–", "5.3"), ("Haiku 5.5", "", "33.8")]
PANELS = [("c36", "(a) $c = 36$, $p = 1$", 44), ("control", "(b) $c = 4$ (true $36$), $p = 1$", 22),
          ("wrong50", "(c) $c = 36$, $p = 0.5$ (true $1$)", 12)]
C = ps.BLUE.edge

val = {}
for r in csv.DictReader(open(DATA)):
    f = lambda k: float(r[k]) if r[k] not in ("", "nan") else math.nan
    val[(r["model"], r["mode"], r["panel"])] = (f("ess"), f("lo"), f("hi"), f("bayes"), f("ref_ess"), f("ref_lo"), f("ref_hi"))

ps.use_style()
fig, axes = ps.figure(ncols=4, width=ps.WIDE, height=2.55, sharey=True,
                      gridspec_kw=dict(width_ratios=[1, 1, 1, 0.68], wspace=0.08))
for ax, (panel, title, xmax) in zip(axes, PANELS):
    bayes = next(v[3] for k, v in val.items() if k[2] == panel)
    ax.axvline(bayes, **ps.BAYES)
    for y, (model, _, _) in enumerate(ROWS):
        pts = []
        for mode, filled, dy, ls in (("without", False, -0.17, (0, (2.2, 1.6))), ("with", True, 0.17, "-")):
            v = val.get((model, mode, panel))
            if v is None:
                continue
            e, lo, hi, _, ref, rlo, rhi = v
            mark = "o"
            if not math.isfinite(e):  # own curve never falls to the error with the description: Bayes-optimal curve
                e, lo, hi, mark = ref, rlo, rhi, "s"
            x = e
            top = xmax if not math.isfinite(hi) else min(hi, xmax)
            ax.plot([lo, top], [y + dy, y + dy], color=C, lw=1, ls=ls, alpha=0.7, zorder=2, solid_capstyle="butt")
            ax.plot(x, y + dy, mark, ms=6, mfc=C if filled else "white", mec=C, mew=1.3, zorder=3, clip_on=False)
            pts.append(x)
    ax.set_xlim(-1.5 * xmax / 22, xmax)
    ax.set_title(title, fontsize=8.6, color=ps.TEXT, loc="left")
    ax.set_xlabel("ESS (demonstrations)")
    ps.style_axes(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(range(len(ROWS)), [r[0] for r in ROWS])
axes[0].set_ylim(len(ROWS) - 0.5, -0.6)
tx = axes[3]
tx.set_xlim(0, 2)
tx.axis("off")
for y, (_, h0, h1) in enumerate(ROWS):
    for x, h in ((0.5, h0), (1.5, h1)):
        tx.text(x, y, h, ha="center", va="center", color=ps.TEXT, fontsize=8.2)
tx.set_title("HLE (%)", fontsize=8.6, color=ps.TEXT, loc="center", pad=19)
tx.text(0.5, -0.6, "without\nreasoning", ha="center", va="bottom", color=ps.MUTED, fontsize=7.4, linespacing=1.0)
tx.text(1.5, -0.6, "with\nreasoning", ha="center", va="bottom", color=ps.MUTED, fontsize=7.4, linespacing=1.0)
axes[0].plot([], [], "o", ms=5, mfc="white", mec=C, mew=1.3, color=C, lw=1, ls=(0, (2.2, 1.6)), label="without reasoning")
axes[0].plot([], [], "o", ms=5, mfc=C, mec=C, mew=1.3, color=C, lw=1, ls="-", label="with reasoning")
axes[0].plot([], [], "s", ms=5, mfc=C, mec=C, mew=1.3, color=C, lw=1, ls="-", label="with reasoning, in reference demonstrations")
axes[0].plot([], [], label="reference", **ps.BAYES)
fig.legend(loc="outside upper center", ncol=2, fontsize=8.2, handlelength=2.6, handletextpad=0.4, columnspacing=1.2)
ps.save(fig, "llm_ess")
