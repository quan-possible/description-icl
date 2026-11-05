"""Draft figure: each model's response to the stated specificity answering at once (open) and after reasoning
(filled), one row per model, ordered by Humanity's Last Exam (HLE) score; the row label gives HLE in each mode.
Values from Table "strength" (appendix); HLE from Artificial Analysis, 31 October 2026."""
import matplotlib.pyplot as plt
import paper_style as ps

# model, HLE at once, HLE reasoning, (response, low, high) at once, the same reasoning
ROWS = [
    ("GPT-6 Sol", "18.4", "–", (0.08, 0.00, 0.16), (1.00, 1.00, 1.00)),
    ("Gemma 4 31B", "11.8", "", (0.38, 0.34, 0.41), None),
    ("Qwen3.5 9B", "9.4", "14.9", (0.12, 0.09, 0.14), (0.99, 0.99, 1.00)),
    ("Gemma 4 12B", "6.3", "", (0.19, 0.16, 0.23), None),
    ("Qwen3 4B Thinking", "", "6.2", None, (1.08, 0.90, 1.27)),
    ("Olmo 3 7B Think", "", "6.0", None, (0.19, 0.05, 0.33)),
    ("gpt-oss-20b", "–", "5.3", (0.02, 0.02, 0.03), (1.03, 0.93, 1.14)),
]
C = ps.BLUE.edge

ps.use_style()
fig, (ax, tx) = ps.figure(ncols=2, height=1.8, gridspec_kw=dict(width_ratios=[2.7, 1.45], wspace=0.02), sharey=True)
ax.axvline(1, **ps.BAYES)
for y, (name, h0, h1, once, reason) in enumerate(ROWS):
    if once and reason:
        ax.plot([once[0], reason[0]], [y, y], color=ps.SPINE, lw=0.9, zorder=1)
    for v, filled in ((once, False), (reason, True)):
        if v:
            ax.errorbar(v[0], y, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="none", ecolor=C, lw=1, alpha=0.6, zorder=2)
            ax.plot(v[0], y, "o", ms=5, mfc=C if filled else "white", mec=C, mew=1.3, zorder=3)
ax.set_yticks(range(len(ROWS)), [r[0] for r in ROWS])
ax.set_ylim(len(ROWS) - 0.5, -0.7)
ax.set_xlim(-0.08, 1.32)
ax.set_xticks([0, 0.5, 1])
ax.set_xlabel("Response to the stated sd")
ax.annotate("Bayes-optimal", (1, -0.7), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", color=ps.MUTED, fontsize=7.2, annotation_clip=False)
ps.style_axes(ax, grid_axis="x")
ax.tick_params(axis="y", length=0)
# HLE (%) in each mode, as a two-column table beside the rows
tx.set_xlim(0, 2)
tx.axis("off")
for y, (_, h0, h1, _, _) in enumerate(ROWS):
    for x, h in ((0.5, h0), (1.5, h1)):
        tx.text(x, y, h, ha="center", va="center", color=ps.TEXT, fontsize=7.4)
tx.text(0.5, -0.7, "once", ha="center", va="bottom", color=ps.MUTED, fontsize=7.2)
tx.text(1.5, -0.7, "reason", ha="center", va="bottom", color=ps.MUTED, fontsize=7.2)
tx.text(1.0, len(ROWS) - 0.2, "HLE (%)", ha="center", va="top", color=ps.MUTED, fontsize=7.2)
ax.plot([], [], "o", ms=5, mfc="white", mec=C, mew=1.3, label="answering at once")
ax.plot([], [], "o", ms=5, mfc=C, mec=C, mew=1.3, label="reasoning first")
fig.legend(loc="outside upper center", ncol=2, handletextpad=0.3, columnspacing=1.2)
ps.save(fig, "reasoning")
