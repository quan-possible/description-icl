# RQ2: worth of an unreliable description

**What it shows.** When a description is sometimes wrong ($p < 1$), extra
precision stops adding single-query worth, and for precise descriptions the
long-horizon ESS can exceed the single-query ESS. Mis-set trust is cheap
unless it is extreme. A wrong description is abandoned after a few examples.

## Setting

| Symbol | Meaning | Values |
| --- | --- | --- |
| $(d,$ `a0`$)$ | dimension, base prior signal-to-noise ratio | (4, 10), (16, 10), (16, 100), (64, 10) |
| `r` $= s_\ell^2/s_0^2$ | fraction of prior variance the description leaves | 0.5, 0.2, 0.05, 0.01, 0.001 |
| $p$ | true reliability | 1, 0.99, 0.9, 0.7, 0.5, 0.3 |
| $q$ | learner's assumed reliability | equal to $p$ in `ess_map.csv`; varied in `trust.csv` |
| $N$ | horizon | 1, 16, 200 |

$\sigma_y = 1$, inputs $x \sim \mathcal{N}(0, I_d)$, regret is log loss
against the oracle that knows $w$, in nats. Three seeds (0, 1, 2); ESS varies
across seeds by at most about 12%.

## Run

```bash
uv run python results/rq2-reliability/run.py   # about 10 minutes
```

## Tables

- `ess_map.csv`: ESS over $p$ and `r` by horizon, calibrated trust.
- `trust.csv`: regret when $q \ne p$, at $d = 16$, `a0` $= 10$.
- `tipping.csv`: examples before the posterior weight on a wrong description
  falls below 1/2 (median and 95th percentile over prompts; $-1$ means not
  within 80 examples).

## Findings

**Reliability caps the worth of precision.** Single-query ESS as `r` goes
from 0.05 to 0.001:

| $d$ | `a0` | $p = 1$ | $p = 0.99$ | $p = 0.9$ |
| --- | --- | --- | --- | --- |
| 4 | 10 | 5.9 → 105.1 | 5.6 → 29.9 | 4.2 → 7.0 |
| 16 | 10 | 16.9 → 116.5 | 16.6 → 64.2 | 14.8 → 23.1 |
| 16 | 100 | 14.8 → 26.4 | 14.7 → 24.0 | 13.8 → 17.9 |
| 64 | 10 | 62.2 → 165.7 | 61.6 → 134.0 | 57.9 → 78.9 |

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
is abandoned is 1 to 3 for `r` $\le 0.05$ at every trust level tested, up to
$q = 0.999999$. Coarse descriptions (`r` $= 0.5$) take up to 23 at $d = 64$.

## Validation

`tests/test_mixture.py` checks the mixture learner. Two further checks agree
with the tables: the long-horizon cost of mis-set trust equals
$\mathrm{KL}(p\,\|\,q)$, and the long-horizon ESS matches the examples whose
information gain is $p\,\tfrac{d}{2}\log(1/r) - H(p)$ (9.2 predicted, 9.22
computed, at $d = 16$, `a0` $= 10$, `r` $= 0.05$, $p = 0.9$).
