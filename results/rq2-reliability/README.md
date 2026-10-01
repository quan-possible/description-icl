# RQ2: worth of an unreliable description

**What it shows.** When a description is sometimes wrong ($p < 1$), extra
precision stops adding single-query worth, and for precise descriptions the
long-horizon ESS can exceed the single-query ESS. Mis-set trust is cheap
unless it is extreme. A wrong description is abandoned after a few examples.

## Setting

| Symbol | Meaning | Values |
| --- | --- | --- |
| $(d,$ `a0`$)$ | dimension, base prior signal-to-noise ratio | (4, 10), (16, 10), (16, 100), (64, 10) |
| `r` $= s_\ell^2/s_0^2$ | fraction of prior variance the description leaves | 0.1, 0.01, 0.001 |
| $p$ | true reliability | 1, 0.99, 0.9 (`trust.csv` also 0.7, 0.5) |
| $q$ | learner's assumed reliability | equal to $p$ in `ess_map.csv`; varied in `trust.csv` |
| $N$ | horizon | 1, 16, 200 |

$\sigma_y = 1$, inputs $x \sim \mathcal{N}(0, I_d)$, regret is log loss
against the oracle that knows $w$, in nats. Three seeds (0, 1, 2); the one-step ESS
varies across seeds by at most 0.5%.

## Run

```bash
uv run python results/rq2-reliability/run.py   # about 5 minutes
uv run python results/rq2-reliability/table.py  # reliability_table.tex, the paper's Table 2
```

## Tables

- `ess_map.csv`: ESS over $p$ and `r` by horizon, calibrated trust.
- `trust.csv`: regret when $q \ne p$, at $d = 16$, `a0` $= 10$.
- `tipping.csv`: examples before the posterior weight on a wrong description
  falls below 1/2 (median and 95th percentile over prompts; $-1$ means not
  within 80 examples).

## Findings

**Reliability caps the worth of precision.** Single-query ESS as `r` goes
from 0.1 to 0.001:

| $d$ | `a0` | $p = 1$ | $p = 0.99$ | $p = 0.9$ |
| --- | --- | --- | --- | --- |
| 4 | 10 | 4.4 → 105.1 | 4.3 → 29.9 | 3.5 → 7.0 |
| 16 | 10 | 14.9 → 116.5 | 14.6 → 64.2 | 13.1 → 23.1 |
| 16 | 100 | 13.8 → 26.3 | 13.6 → 24.0 | 12.6 → 17.9 |
| 64 | 10 | 57.8 → 165.7 | 57.2 → 134.0 | 53.0 → 78.9 |

**The horizon ordering reverses in three of four settings.** ESS at
`r` $= 0.001$, $p = 0.9$:

| $d$ | `a0` | $N = 1$ | $N = 200$ |
| --- | --- | --- | --- |
| 4 | 10 | 7.0 | 37.4 |
| 16 | 10 | 23.1 | 49.8 |
| 64 | 10 | 78.9 | 82.7 |
| 16 | 100 | 17.9 | 15.4 |

The boundary of the reversal is not yet derived.

**Mis-set trust is cheap unless extreme.** Single-query regret at $d = 16$,
`a0` $= 10$, `r` $= 0.001$, $p = 0.7$:

| $q$ | 0 | 0.1 | 0.5 | 0.7 | 0.9 | 0.99 | 0.999 | 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regret | 2.51 | 1.95 | 1.33 | 1.27 | 1.39 | 1.90 | 2.44 | 40.5 |

At $N = 200$ the excess over calibrated trust matches
$\mathrm{KL}(p\,\|\,q)$: 1.46 nats at $q = 0.999$.

**Tipping is fast.** The median number of examples before a wrong description
is abandoned is 1 or 2 for `r` $\le 0.01$ at every trust level tested, up to
$q = 0.999999$, and 1 to 5 for `r` $= 0.1$; at $d = 4$, `r` $= 0.1$ the
95th percentile is beyond 80 examples once $q \ge 0.9$, because a coarse
wrong description is nearly compatible with the data.

## Worth with examples in hand

`worth.py` computes the description's worth as further examples saved when
$n$ examples are already present (paper, Definition 2 and Proposition 3) at
$d = 16$, `a0` $= 10$, three seeds, into `worth.csv`; `worth_figure.py`
draws `worth.pdf` for the paper with the network panel from
`results/rq3-meta-trained/`. A precise unreliable description gains worth
as examples verify it (23 alone, 30 after one, 29 after ten, 88 after a
hundred at `r` $= 0.001$, $p = 0.9$); a reliable one drifts from 117 to
108. The grid is $n = 0, \dots, 100$ with 8,000 prompts per seed; the table
reports $n \in \{0, 1, 10, 100\}$.

## Validation

`tests/test_mixture.py` checks the mixture learner. Two further checks agree
with the tables: the long-horizon cost of mis-set trust equals
$\mathrm{KL}(p\,\|\,q)$, and the long-horizon ESS matches the examples whose
information gain is $p\,\tfrac{d}{2}\log(1/r) - H(p)$ (9.2 predicted, 9.22
computed, at $d = 16$, `a0` $= 10$, `r` $= 0.05$, $p = 0.9$).
