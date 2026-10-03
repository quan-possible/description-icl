# How Many Demonstrations Is a Task Description Worth?

The paper and everything it is built from. A task description's *effective sample size* (ESS) is the
number of demonstrations that lower a predictor's regret as much as the description does; the paper
computes it for the Bayes-optimal predictor in in-context linear regression, then measures it for
meta-trained Transformers and pretrained language models.

## Layout

| Path | What it holds |
| --- | --- |
| `docs/paper/` | `paper.tex` (ICML 2026 style), `refs.bib`, the built `paper.pdf`, and `paper.md` (a Markdown copy, `to_md.sh`) |
| `docs/paper/figs/` | Figure scripts and their outputs; `paper_style.py` is the shared figure style |
| `results/rq3-meta-trained/` | The meta-trained Transformers of Section 6 (Figure 3, Table 1) |
| `results/llm-test/` | The language-model test of Section 7 (Figure 4, Tables 2–7); see its `README.md` |
| `src/description_icl/`, `tests/` | The library behind the meta-trained networks |

## Build

```bash
uv sync
cd docs/paper/figs && uv run python ess_curve.py && uv run python ess_figure.py && uv run python networks_figure.py
cd ../../../results/llm-test && uv run python analyze.py replot
cd ../../docs/paper && latexmk -pdf paper.tex
```

`ess_curve.py` recomputes the Section 4–5 numbers (minutes); the others only redraw. `analyze.py replot`
redraws Figure 4 and the language-model tables from the last analysis; run `analyze.py` on the
`full_*.npz` score files to recompute them. Large pilot arrays (`results/llm-test/archive-v2/*.npz`)
are kept out of Git.
