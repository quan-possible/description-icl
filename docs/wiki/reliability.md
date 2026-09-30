# Reliability: the regret floor and saturation

**Status:** computed in four settings; the floor has a two-line sketch and
needs a write-up. In the paper.

**Result.** With no examples the Bayes-optimal predictive is the $p$-mixture
of the two components' predictives. As $r \to 0$ its one-step regret tends to

$$(1-p)\,R^{\mathrm{ex}}(0) + H(p),$$

where $R^{\mathrm{ex}}(0)$ is the regret with neither description nor
examples and $H$ is the binary entropy. So the ESS of an unreliable
description saturates in precision at the $n$ where $R^{\mathrm{ex}}(n)$
reaches the floor.

**Why.** As $r \to 0$ the components separate. Correct description: the
mixture gives the label $p$ times the oracle's density, cost $\log(1/p)$.
Wrong description: it gives $(1-p)$ times the base predictive's density,
cost $R^{\mathrm{ex}}(0) + \log(1/(1-p))$. Weight by $p$ and $1 - p$.

**Check** ($d = 16$, $a_0 = 10$, $R^{\mathrm{ex}}(0) = 2.51$ nats):

| $p$ | Formula | Computed at $r = 0.001$ |
| --- | --- | --- |
| 0.9 | 0.58 | 0.57 |
| 0.7 | 1.36 | 1.27 |
| 0.5 | 1.95 | 1.80 |

The limit is approached from above as $r \to 0$.

**Numbers** (one-step ESS, $d = 16$, $a_0 = 10$, calibrated predictor, from
`results/rq2-reliability/ess_map.csv`):

| $r$ | $p = 1$ | $0.99$ | $0.9$ | $0.7$ |
| --- | --- | --- | --- | --- |
| 0.5 | 7.6 | 7.5 | 6.3 | 4.1 |
| 0.05 | 16.9 | 16.6 | 14.8 | 11.7 |
| 0.001 | 116.5 | 64.2 | 23.1 | 15.6 |

Holds at $(d, a_0) \in \{(4, 10), (16, 10), (16, 100), (64, 10)\}$. ESS
varies across seeds by at most 12%.

**Under squared error** the saturation is stronger: at $d = 5$, $r = 0.05$,
ESS is 7.4 at $p = 1$ and 4.4 at $p = 0.9$ (`results/rq3-meta-trained/targets.csv`).
