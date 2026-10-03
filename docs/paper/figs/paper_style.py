"""The paper's figure style: the Editorial Neutral chart binding (editorial_data_viz.py, vendored from
the visualization-design skill) at print size, in Inter to match Figure 1.

Color roles, kept the same in every figure:
  green  a reliable description (relevant with probability 1)
  pink   a description that may be irrelevant
  blue   a single series without either role (a network, a model)
  ink, dashed  the Bayes-optimal predictor
Everything else is gray.
"""

import glob
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

import editorial_data_viz as ed
from editorial_data_viz import BLUE, GRAY, GREEN, MUTED, PINK, SPINE, SUBTLE, TEXT, soften_bar, style_axes  # noqa: F401

COLUMN, WIDE = 3.25, 6.75  # inches: one column, both columns of the ICML page
BAYES = dict(color=MUTED, lw=1.3, ls=(0, (4, 3)))  # the Bayes-optimal predictor, in every figure

for path in glob.glob("/usr/local/texlive/*/texmf-dist/fonts/opentype/public/inter/Inter-*.otf"):
    font_manager.fontManager.addfont(path)


def use_style():
    """The binding's rc settings, with its screen sizes (tick 10, label 10.5, legend 9.5 px) scaled to print points."""
    ed.use_diagram_chart_style()
    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 8.4, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.6,
        "mathtext.fontset": "custom", "mathtext.rm": "Inter", "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
        "pdf.fonttype": 42, "legend.frameon": False, "axes.unicode_minus": True,
        "lines.solid_capstyle": "round", "errorbar.capsize": 0,
    })


def figure(ncols=1, width=COLUMN, height=2.1, **kw):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), layout="constrained", **kw)
    return fig, axes


def panel_labels(axes, labels="abcdefg", titles=()):
    """(a), (b), ... at each panel's top-left, in ink at label size, quiet rather than bold; an optional short title follows in muted ink."""
    titles = list(titles) + [""] * len(axes)
    for ax, l, t in zip(axes, labels, titles):
        a = ax.annotate(f"({l})", (0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points", ha="left", va="bottom", color=TEXT, fontsize=8.4)
        if t:
            ax.annotate(t, (1, 0), xycoords=a, xytext=(4, 0), textcoords="offset points", ha="left", va="bottom", color=MUTED, fontsize=8.4)


def save(fig, stem):
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.svg")
    plt.close(fig)
