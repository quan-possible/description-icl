# RQ1: single-query versus long-horizon ESS of a reliable description

**What it shows.** For a reliable description ($p = 1$), the single-query ESS
always exceeds the long-horizon ESS. Their ratio grows with dimension and
signal-to-noise ratio and shrinks as the description becomes more precise.
It ranges from 0.99 to 3.3 over this grid (the earlier five-value grid
reached 8.5 at $r = 0.9$, $a_0 = 100$, so "up to three times" holds for
$r \le 0.1$ only). `table.py` writes `precision_table.tex`, the body of the
paper's Table 1, from `ess.csv`.

## Setting

| Symbol | Meaning | Values |
| --- | --- | --- |
| $d$ | dimension | 1, 2, 4, 8, 16, 32, 64 |
| `a0` $= s_0^2/\sigma_y^2$ | base prior signal-to-noise ratio | 1, 10, 100 |
| `r` $= s_\ell^2/s_0^2$ | fraction of prior variance the description leaves | 0.1, 0.01, 0.001 |
| $N$ | horizon | 1, 4, 16, 64, 256, $\infty$ |

Inputs are $x \sim \mathcal{N}(0, I_d)$. Regret is log loss against the oracle
that knows $w$, in nats. Three seeds (0, 1, 2); `_sd` columns give the
standard deviation of the ESS across seeds.

## Run

```bash
uv run python results/rq1-single-query-gap/run.py   # about 15 minutes
uv run python results/rq1-single-query-gap/table.py
```

## Columns of `ess.csv`

- `ess_N<k>`: ESS at horizon $k$, linearly interpolated between integers.
- `ess_inf`: long-horizon limit, the examples whose expected information gain
  equals the description's, $\tfrac{d}{2}\log(1/r)$.
- `ess_sq`: single-query ESS under squared error.
- `ratio_1_inf`: `ess_N1 / ess_inf`.
- `ratio_approx`: $(1-r)\log(1 + d\,a_0)/\log(1/r)$, an approximation that
  assumes high SNR and fewer than $d$ examples.
- `balanced_ess`: ESS under balanced designs ($X^\top X = nI$), where it is
  the same at every horizon: $1/(r a_0) - 1/a_0$.

## Findings

| $d$ | `a0` | `r` | single-query ESS | long-horizon ESS | ratio |
| --- | --- | --- | --- | --- | --- |
| 1 | 10 | 0.1 | 1.95 | 1.66 | 1.17 |
| 16 | 10 | 0.1 | 14.9 | 7.76 | 1.92 |
| 64 | 10 | 0.1 | 57.7 | 23.6 | 2.44 |
| 64 | 100 | 0.1 | 56.9 | 17.1 | 3.32 |
| 64 | 100 | 0.01 | 64.0 | 35.1 | 1.82 |
| 64 | 1 | 0.01 | 163.0 | 132.8 | 1.23 |
| 16 | 10 | 0.001 | 116.8 | 108.5 | 1.08 |

- The gap is geometric. In $d = 1$ the ratio stays between 0.99 and 1.22;
  under balanced designs it is exactly 1.
- For coarse descriptions at high SNR the single-query ESS is close to
  $d(1 - r)$: a description that removes a fraction $1 - r$ of the prior
  variance is worth that fraction of $d$ examples.
- In the proportional limit ($d, n \to \infty$, $d\,a_0$ fixed) both ESS
  values have closed forms from the Marchenko–Pastur law
  (`gaussian.ess_proportional`). At $d = 64$ they match the simulation to
  within about 2%.

## Validation

`tests/test_gaussian.py` checks the routine against a direct
log-determinant, $d = 1$ quadrature, the high-SNR Wishart limit, balanced
designs, the long-horizon limit, and the proportional limit.
