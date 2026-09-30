# Reliability: the regret floor and saturation

**Status:** computed in four settings; the floor has a two-line sketch and
needs a write-up. In the paper.

**Result.** With no examples the Bayes-optimal predictive is the $p$-mixture
of the two components' predictives. Its one-step regret satisfies, for every
$r$,

$$R \ge (1-p)\,R^{\mathrm{ex}}(0),$$

and as $r \to 0$ it lies between that bound and
$(1-p)\,R^{\mathrm{ex}}(0) + H(p)$. So the ESS of an unreliable description
is capped at the $n$ where $R^{\mathrm{ex}}(n)$ reaches the lower bound,
however precise the description.

**Why.** Lower bound: a predictor told which component is true has regret
$p R^{\mathrm{desc}}_{p=1} + (1-p) R^{\mathrm{ex}}(0)$, and conditioning on
more information cannot raise a Bayes-optimal predictor's expected log loss.
Upper limit: the mixture density is at least each weighted component, so a
correct description costs at most $\log(1/p)$ as $r \to 0$ and a wrong one
at most $R^{\mathrm{ex}}(0) + \log(1/(1-p))$; weighting gives
$(1-p) R^{\mathrm{ex}}(0) + H(p)$.

**Check** ($d = 16$, $a_0 = 10$, $R^{\mathrm{ex}}(0) = 2.51$ nats, computed at
$r = 0.001$):

| $p$ | Lower bound | Upper limit | Computed |
| --- | --- | --- | --- |
| 0.9 | 0.25 | 0.58 | 0.57 |
| 0.7 | 0.75 | 1.36 | 1.27 |
| 0.5 | 1.26 | 1.95 | 1.80 |

The earlier statement that the regret "tends to" the upper expression from
above was wrong: the computed values sit just below it.

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
