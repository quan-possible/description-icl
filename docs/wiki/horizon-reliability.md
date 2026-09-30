# Horizon and reliability together: the reversal

**Status:** computed in four settings; the asymptotic formula is derived
informally and checked at one point; the boundary is not derived. Written up
in the archived horizon draft.

**Asymptotic ESS.** Combining the regret decomposition (Gaussian term plus
identification term) with the long-horizon limit: as $N \to \infty$ the
description is worth the examples whose information gain is

$$p\,\tfrac d2 \log(1/r) - H(p).$$

Check: $d = 16$, $a_0 = 10$, $r = 0.05$, $p = 0.9$: predicted 9.2, computed
9.22. Unbounded in precision, unlike the one-step ESS
([reliability.md](reliability.md)).

**The reversal.** For precise, unreliable descriptions the asymptotic ESS
exceeds the one-step ESS, the opposite of [horizon.md](horizon.md): feedback
identifies the mixture component, after which the predictor exploits the
description's full precision.

| $d$ | $a_0$ | $N = 1$ | $N = 200$ | Reversed |
| --- | --- | --- | --- | --- |
| 4 | 10 | 7.0 | 37.4 | yes |
| 16 | 10 | 23.1 | 49.8 | yes |
| 64 | 10 | 78.9 | 82.7 | barely |
| 16 | 100 | 17.9 | 15.4 | no |

($r = 0.001$, $p = 0.9$.) Conditional on the setting; the boundary should
follow from the asymptotic formula and the one-step floor.

**Regret decomposition.** For a learner with assumed reliability $q$,
$R_N = \mathbb{E}[\tfrac12 \sum \text{information gains of the true component}] + \mathbb{E}[\log \pi_c(n+N) - \log \pi_c(n)]$,
with $\pi_c$ the posterior weight on the true component. Validated in
`tests/test_mixture.py`.

**Why it matters.** The venue assessment judged this the most novel result
in the project. It returns to the paper as one section once the simple
results exist.
