# Precision: ESS of a reliable description

**Status:** computed on the RQ1 grid; derived with its error term in the
paper (Proposition 1). In the paper.

**Result.** For $p = 1$, one prediction, high signal-to-noise ratio, and
$\mathrm{ESS} < d$:

$$\mathrm{ESS}_1 \approx d\,(1 - r).$$

A description that removes a fraction $1 - r$ of the prior variance is worth
that fraction of the dimension in examples.

**Why.** With $n < d$ examples the posterior variance is near zero along the
$n$ observed directions and $a_0$ along the other $d - n$, so as
$a_0 \to \infty$ the one-step regret is
$\tfrac12 \log a_0 + \tfrac12 \mathbb{E}\log\chi^2_{d-n} + o(1)$. The
description contracts every direction to $r a_0$, giving
$\tfrac12 \log(r a_0) + \tfrac12 \mathbb{E}\log\chi^2_{d} + o(1)$. With
$\mathbb{E}\log\chi^2_k = \psi(k/2) + \log 2$ the ESS solves
$\psi(\tfrac{d-n}{2}) - \psi(\tfrac d2) = \log r$, and expanding the
digamma gives

$$\mathrm{ESS}_1 = (d-1)(1-r) + O\!\left(\tfrac{1}{rd}\right).$$

The error term says the rule needs $rd \gg 1$: the description must leave
well over one example's worth of variance unexplained. At $a_0 = 100$ the
exact ESS is within 0.1 of $(d-1)(1-r)$ for $d \in \{16, 64\}$,
$r \in \{0.5, 0.2\}$ (7.50, 12.1, 31.6, 50.5 against 7.5, 12.0, 31.5,
50.4).

**Numbers** ($a_0 = 10$, from `results/rq1-single-query-gap/ess.csv`):

| $d$ | $r$ | ESS | $d(1-r)$ |
| --- | --- | --- | --- |
| 4 | 0.5 | 1.77 | 2.0 |
| 16 | 0.5 | 7.62 | 8.0 |
| 16 | 0.05 | 16.9 | 15.2 |
| 16 | 0.01 | 26.1 | 15.8 |
| 64 | 0.5 | 31.6 | 32.0 |
| 64 | 0.05 | 62.1 | 60.8 |
| 64 | 0.01 | 73.5 | 63.4 |

Past $\mathrm{ESS} = d$ the approximation stops and the exact value keeps
growing, because $d$ noisy examples do not determine $w$.

**Exact form.** With $G_a(k) = \tfrac12 \mathbb{E} \log\det(I + a X_k^\top X_k)$,
the description's one-step regret is $G_{r a_0}(1)$ and the examples' is
$G_{a_0}(n+1) - G_{a_0}(n)$. Balanced designs ($X^\top X = n I$) give
$\mathrm{ESS} = 1/(r a_0) - 1/a_0$ exactly at every horizon.

**Under squared error** the same quantities use $\operatorname{tr}\Sigma$:
`gaussian.ess_squared`; column `ess_sq` of the RQ1 table. Values are 10 to
20% higher than under log loss.
