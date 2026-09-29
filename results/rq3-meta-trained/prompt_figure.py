"""Draw prompt.svg: how one RQ3 prompt is generated, what the model reads,
and how its predictions are scored. The example prompt comes from
meta.sample_batch, so the figure stays in step with the code.

    uv run python results/rq3-meta-trained/prompt_figure.py
"""

import math
import pathlib

import torch

from descriptor_icl import meta

R, A0, P = 0.05, 10.0, 0.7  # precision, base SNR, reliability of the illustrated model
gen = torch.Generator().manual_seed(3)
b = meta.sample_batch(2, 3, 2, A0, torch.tensor([R]), 1.0, 1.0, "cpu", gen)
tok = meta.tokens(b)[0].tolist()
m, w = b["m"][0].tolist(), b["w"][0].tolist()
Y = b["Y"][0].tolist()

f2 = lambda v: f"{v:.2f}".replace("-", "−")
out = []
add = out.append

CSS = """
svg { background: #ffffff; }
text { font-family: Inter, "Clear Sans", "Noto Sans", "Helvetica Neue", Arial, sans-serif; fill: #171717; }
text.muted { fill: #5a5a5a; } text.subtle { fill: #8a8a8a; }
text.mono { font-family: "DejaVu Sans Mono", "Noto Mono", Menlo, monospace; }
.box-gray { fill: #f5f5f5; stroke: #b3b3b3; stroke-width: 1.3; }
.box-green { fill: #edf3e8; stroke: #8a9f7a; stroke-width: 1.3; }
.box-pink { fill: #f5e7ec; stroke: #bd7c8f; stroke-width: 1.3; }
.box-blue { fill: #e8f0f9; stroke: #7f9fc4; stroke-width: 1.3; }
.region { fill: #fafafa; stroke: #ececec; stroke-width: 1; }
.path-gray { fill: none; stroke: #a8a8a8; stroke-width: 2.4; stroke-linecap: round; }
.path-pink { fill: none; stroke: #bd7c8f; stroke-width: 2.4; stroke-linecap: round; }
.path-green { fill: none; stroke: #8a9f7a; stroke-width: 2.4; stroke-linecap: round; }
.path-weak { stroke-dasharray: 4 3; }
.axis { stroke: #a8a8a8; stroke-width: 0.9; }
"""


def text(x, y, s, size=15, cls="", anchor="start"):
    c = f' class="{cls}"' if cls else ""
    add(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"{c}>{s}</text>')


def box(x, y, w, h, cls, lines=(), mono=False):
    """lines: (text, size, class) tuples, centered and stacked."""
    add(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="7"/>')
    pitch = [s + 5 for _, s, _ in lines]
    top = y + h / 2 - sum(pitch) / 2
    for (s, size, c), p in zip(lines, pitch):
        top += p
        text(x + w / 2, top - 5, s, size, (c + (" mono" if mono else "")).strip(), "middle")


def arrow(d, color="gray", weak=False):
    extra = " path-weak" if weak else ""
    add(f'<path class="path-{color}{extra}" d="{d}" marker-end="url(#arrow-{color})"/>')


def region(y, h, title):
    add(f'<rect class="region" x="20" y="{y}" width="900" height="{h}" rx="13"/>')
    text(40, y + 30, title, 16)


W, H = 940, 1080
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add(f"<style>{CSS}</style><defs>")
for name, col in (("gray", "#a8a8a8"), ("pink", "#bd7c8f"), ("green", "#8a9f7a")):
    add(f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" '
        f'markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{col}"/></marker>')
add("</defs>")
add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

text(20, 34, "How one RQ3 prompt is built, read, and scored", 18)
text(20, 56, "Example with d = 2 and 3 examples, drawn by the project code (meta.sample_batch, seed 3).",
     13, "muted")

# 1 · generation
region(74, 176, "1 · Draw a task and its description")
bx, by, bw, bh = 40, 124, 190, 104
box(bx, by, bw, bh, "box-green", [("Description", 15, ""), (f"m = ({f2(m[0])}, {f2(m[1])})", 13, "muted"),
                                   ("one vector about w", 13, "muted")])
box(262, by, bw, bh, "box-gray", [("Is it correct?", 15, ""), ("yes with probability p", 13, "muted"),
                                   ("here: yes", 13, "muted")])
box(484, by, bw, bh, "box-gray", [("Task weights w", 15, ""), ("near m if correct,", 13, "muted"),
                                   ("unrelated if not", 13, "muted"),
                                   (f"here ({f2(w[0])}, {f2(w[1])})", 13, "muted")])
box(706, by, 196, 46, "box-blue", [("Inputs x_k: random", 14, "")])
box(706, by + 58, 196, 46, "box-pink", [("Answers y_k = w · x_k + noise", 13, "")])
arrow(f"M 804 {by + 46} V {by + 58}")
for x0 in (230, 452, 674):
    arrow(f"M {x0} {by + bh / 2} H {x0 + 32}")

# 2 · the matrix
region(270, 432, "2 · The matrix the model reads: one row (token) per step")
cols = [(130, 86), (222, 86), (330, 110), (462, 90)]
hy = 332
text(219, hy, "vector", 13, "muted", "middle")
text(219, hy + 17, "m in row 0, input x_k after", 11.5, "subtle", "middle")
text(385, hy, "previous answer", 13, "muted", "middle")
text(385, hy + 17, "y_(k−1)", 11.5, "subtle", "middle")
text(507, hy, "has description", 13, "muted", "middle")
text(507, hy + 17, "1 or 0", 11.5, "subtle", "middle")
text(750, hy, "the model predicts", 13, "muted", "middle")
text(750, hy + 17, "a distribution for the next answer", 11.5, "subtle", "middle")

top0, pitch, ch = 366, 62, 40
for k in range(4):
    y = top0 + k * pitch
    text(40, y + 26, f"token {k}", 14, "muted")
    for j, (x, wc) in enumerate(cols):
        v = tok[k][j]
        if k == 0:
            cls = "box-green" if j < 2 else "box-gray"
        else:
            cls = "box-blue" if j < 2 else ("box-pink" if j == 2 and k > 1 else "box-gray")
        box(x, y, wc, ch, cls, [(f2(v) if j < 3 else str(int(v)), 14, "")], mono=True)
    arrow(f"M 552 {y + ch / 2} H 598")
    if k == 0:
        box(600, y, 300, ch, "box-gray", [("nothing yet: reads the description", 13, "muted")])
    else:
        box(600, y, 300, ch, "box-pink", [(f"y{k} = ?      (true value {f2(Y[k - 1])})", 14, "")])
    if 1 <= k <= 2:  # the answer is revealed in the next row
        yn = top0 + (k + 1) * pitch
        arrow(f"M 630 {y + ch} V {yn - 11} H 385 V {yn}", "pink")

ny = top0 + 3 * pitch + ch + 32
text(40, ny, "Pink arrows: each answer is revealed in the next row, so row k shows the model rows 0 to k "
     "and never the answer it predicts.", 13, "muted")
text(40, ny + 21, "Without a description, row 0 is all zeros and rows 1–3 are unchanged: the two prompts "
     "differ only in row 0.", 13, "muted")
text(40, ny + 42, "Not in the matrix: precision r and reliability p. Each model is trained at one value of "
     "each and learns them from the prompts.", 13, "muted")
text(40, ny + 63, "Real runs: d = 8 and 32 examples, so 33 rows of d + 2 = 10 numbers.", 13, "muted")

# 3 · output and scoring
y3 = 722
region(y3, 334, "3 · What comes out, and how it is scored")
px0, px1, pbase, ph = 64, 392, y3 + 222, 118
lo, hi = -5.0, 7.0
X = lambda v: px0 + (v - lo) / (hi - lo) * (px1 - px0)
x1 = b["X"][0, 0].tolist()
sq = x1[0] ** 2 + x1[1] ** 2
comps = [(P, m[0] * x1[0] + m[1] * x1[1], 1 + R * A0 * sq), (1 - P, 0.0, 1 + A0 * sq)]
dens = lambda v, c: c[0] * math.exp(-0.5 * (v - c[1]) ** 2 / c[2]) / math.sqrt(2 * math.pi * c[2])
grid = [lo + i * (hi - lo) / 240 for i in range(241)]
top = max(dens(v, comps[0]) + dens(v, comps[1]) for v in grid)
Yp = lambda d: pbase - d / top * ph
curve = lambda f: "M " + " L ".join(f"{X(v):.1f} {Yp(f(v)):.1f}" for v in grid)
text(40, y3 + 60, "Ideal prediction for y1 in row 1", 14)
text(40, y3 + 79, "for a model trained at p = 0.7 and r = 0.05", 13, "muted")
add(f'<line class="axis" x1="{px0}" y1="{pbase}" x2="{px1}" y2="{pbase}"/>')
for t in range(-4, 7, 2):
    add(f'<line class="axis" x1="{X(t):.1f}" y1="{pbase}" x2="{X(t):.1f}" y2="{pbase + 4}"/>')
    text(X(t), pbase + 18, f2(t)[:-3], 11, "muted", "middle")
add(f'<path class="path-green path-weak" style="stroke-width:1.6" d="{curve(lambda v: dens(v, comps[0]))}"/>')
add(f'<path class="path-gray path-weak" style="stroke-width:1.6" d="{curve(lambda v: dens(v, comps[1]))}"/>')
add(f'<path class="path-pink" style="stroke-width:2.2" d="{curve(lambda v: dens(v, comps[0]) + dens(v, comps[1]))}"/>')
xt = X(Y[0])
add(f'<line x1="{xt:.1f}" y1="{pbase}" x2="{xt:.1f}" y2="{pbase - ph - 4}" stroke="#171717" stroke-width="1.2"/>')
text(xt - 6, pbase - 0.72 * ph, f"true y1 = {f2(Y[0])}", 12, "", "end")
ly = pbase + 42
for i, (lab, col, dash) in enumerate((("weight 0.7: description right", "#8a9f7a", "4 3"),
                                       ("weight 0.3: description wrong", "#a8a8a8", "4 3"),
                                       ("their sum: the prediction", "#bd7c8f", ""))):
    yy = ly + i * 18
    add(f'<line x1="64" y1="{yy}" x2="88" y2="{yy}" stroke="{col}" stroke-width="2" stroke-dasharray="{dash}"/>')
    text(96, yy + 4, lab, 11.5, "muted")

b1 = y3 + 48
box(440, b1, 210, 58, "box-gray", [("Trained Transformer", 15, ""), ("two bell curves per row", 13, "muted")])
box(690, b1, 210, 58, "box-gray", [("Ideal learner", 15, ""), ("exact formula, same prompts", 13, "muted")])
arrow(f"M 545 {b1 + 58} V {b1 + 84}")
arrow(f"M 795 {b1 + 58} V {b1 + 84}")
box(440, b1 + 86, 460, 50, "box-gray", [("Regret per prediction", 15, ""),
                                         ("its log loss minus that of an oracle who knows w", 13, "muted")])
arrow(f"M 670 {b1 + 136} V {b1 + 158}")
box(440, b1 + 160, 460, 74, "box-gray", [("ESS of the description", 15, ""),
                                          ("examples, without a description, that give", 13, "muted"),
                                          ("the same regret as the description alone", 13, "muted")])
text(670, b1 + 262, "Question: is the network's ESS the same as the ideal learner's?", 13, "muted", "middle")

add("</svg>")
path = pathlib.Path(__file__).with_name("prompt.svg")
path.write_text("\n".join(out))
print(path)
