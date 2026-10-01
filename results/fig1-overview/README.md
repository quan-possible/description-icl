# Figure 1: overview

**What it shows.** A description is worth a number of in-context examples,
set by its precision and capped by its reliability. (a) is a schematic of the
ESS definition and the robust-mixture prior. (b) shows the definition on one
description ($r = 0.001$): ESS 116 at $p = 1$, 23 at $p = 0.9$. (c) shows ESS
against precision $1/r$ for $p \in \{1, 0.99, 0.9, 0.7\}$.

## Setting

$d = 16$, `a0` $= 10$, $\sigma_y = 1$, single query ($N = 1$), log loss in
nats. Description regrets and ESS values come from
`results/rq2-reliability/ess_map.csv` (three seeds). The examples-only regret
curve in (b) is recomputed with seeds 100–102; the script asserts that it
crosses the description regrets within 3% of the tabulated ESS (116.6 vs
116.5, 23.1 vs 23.1).

## Run

```bash
uv run python results/fig1-overview/overview.py   # about 10 seconds
```

Writes `overview.pdf` (included by `docs/paper/paper.tex`) and `overview.svg`,
built at 5.5 in, the NeurIPS text width. `orx_figstyle.py` is the vendored
figure style.
