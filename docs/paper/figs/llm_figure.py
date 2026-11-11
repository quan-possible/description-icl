"""§6 figure: each language model's ESS without reasoning (open) and with it (filled), on the same 200 tasks, for the
c = 4 description, the c = 36 description, and the c = 36 description when it is irrelevant one time in ten and the
prompt says so (the matched design of results/llm-test/stated-matched; only the models run in it appear). Dashed:
the Bayes-optimal ESS. ESS from each model's own no-description curve (Definition 1); a square, against a Bayes-optimal learner where the
model's own curve never falls to its error with the description.
Data: results/llm-test/figure_data.csv (figure_data.py), which also holds the stated-50% condition (appendix only)."""
import csv
import math
import pathlib

import paper_style as ps

DATA = pathlib.Path(__file__).resolve().parents[3] / "results/llm-test/figure_data.csv"
ROWS = ["Opus 4.6", "GPT-6 Sol", "Gemma 4 31B", "Qwen3.5 9B", "gpt-oss-20b", "Haiku 5.5"]
PANELS = [("c4", "(a) stated s.d. 30 ($c = 4$)", 8), ("c36", "(b) stated s.d. 10 ($c = 36$)", 44), ("matched10", "(c) s.d. 10, $10\\%$ irrelevant", 8)]  # (c) on (a)'s scale, so that models can be told apart near the reference
C = ps.BLUE.edge

val = {}
for r in csv.DictReader(open(DATA)):
    f = lambda k: float(r[k]) if r[k] not in ("", "nan") else math.nan
    val[(r["model"], r["mode"], r["panel"])] = (f("ess"), f("lo"), f("hi"), f("bayes"), f("ref_ess"), f("ref_lo"), f("ref_hi"))

ps.use_style()
fig, axes = ps.figure(ncols=3, width=ps.WIDE, height=2.4, sharey=True, gridspec_kw=dict(wspace=0.06))
for ax, (panel, title, xmax) in zip(axes, PANELS):
    bayes = next(v[3] for k, v in val.items() if k[2] == panel)
    ax.axvline(bayes, **ps.BAYES)
    for y, model in enumerate(ROWS):
        for mode, filled, dy, ls in (("without", False, -0.17, (0, (2.2, 1.6))), ("with", True, 0.17, "-")):
            v = val.get((model, mode, panel))
            if v is None:
                continue
            e, lo, hi, _, ref, rlo, rhi = v
            mark = "o"
            if not math.isfinite(e):  # own curve never falls to the error with the description: Bayes-optimal curve
                e, lo, hi, mark = ref, rlo, rhi, "s"
            top = xmax if not math.isfinite(hi) else min(hi, xmax)
            ax.plot([lo, top], [y + dy, y + dy], color=C, lw=1, ls=ls, alpha=0.7, zorder=2, solid_capstyle="butt")
            ax.plot(e, y + dy, mark, ms=6, mfc=C if filled else "white", mec=C, mew=1.3, zorder=3, clip_on=False)
    ax.set_xlim(-1.5 * xmax / 22, xmax)
    ax.set_title(title, fontsize=8.6, color=ps.TEXT, loc="left")
    ax.set_xlabel("ESS (demonstrations)")
    ps.style_axes(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(range(len(ROWS)), ROWS)
axes[0].set_ylim(len(ROWS) - 0.5, -0.6)
axes[0].plot([], [], "o", ms=5, mfc="white", mec=C, mew=1.3, color=C, lw=1, ls=(0, (2.2, 1.6)), label="without reasoning")
axes[0].plot([], [], "o", ms=5, mfc=C, mec=C, mew=1.3, color=C, lw=1, ls="-", label="with reasoning")
axes[0].plot([], [], "s", ms=5, mfc=C, mec=C, mew=1.3, color=C, lw=1, ls="-", label="on the reference's baseline")
axes[0].plot([], [], label="reference", **ps.BAYES)
fig.legend(loc="outside upper center", ncol=4, fontsize=8.2, handlelength=2.6, handletextpad=0.4, columnspacing=1.2)
ps.save(fig, "llm_ess")
