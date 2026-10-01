# Reliability: the regret floor and saturation

**Status:** computed in four settings; the two-sided bound is proved in the
paper (Proposition 2). In the paper.

**Result.** With no examples the Bayes-optimal predictive is the $p$-mixture
of the two components' predictives. Its one-step regret satisfies, for every
$p$ and $r$,

$$p\,R^{\mathrm{desc}}_{p=1}(r) + (1-p)\,R^{\mathrm{ex}}(0) \;\le\; R \;\le\;
p\,R^{\mathrm{desc}}_{p=1}(r) + (1-p)\,R^{\mathrm{ex}}(0) + H(p).$$

The gap is the cost of not knowing whether the description is right, at most
$H(p)$. Since $R^{\mathrm{desc}}_{p=1}(r) \ge 0$ and $\to 0$ as $r \to 0$,
$R \ge (1-p)\,R^{\mathrm{ex}}(0)$ for every $r$, and as $r \to 0$ the
regret lies between that floor and $(1-p)\,R^{\mathrm{ex}}(0) + H(p)$. So
the ESS of an unreliable description is capped at the $n$ where
$R^{\mathrm{ex}}(n)$ reaches the floor, however precise the description.

**Why.** Lower bound: given whether the description is correct, the true
conditional density of $y$ is one component's predictive, and the mixture's
expected log loss exceeds it by a KL divergence; the two components' regrets
are $R^{\mathrm{desc}}_{p=1}(r)$ and $R^{\mathrm{ex}}(0)$. Upper bound: the
mixture density is at least $p$ times the correct component and $(1-p)$
times the base predictive, so the excess over the informed predictor is at
most $\log(1/p)$ when the description is correct and $\log(1/(1-p))$ when
it is wrong; weighting gives $H(p)$.

**Check** ($d = 16$, $a_0 = 10$, $R^{\mathrm{ex}}(0) = 2.51$ nats,
$R^{\mathrm{desc}}_{p=1}(0.001) = 0.07$, computed at $r = 0.001$):

| $p$ | Lower bound | Upper bound | Computed |
| --- | --- | --- | --- |
| 0.9 | 0.32 | 0.64 | 0.57 |
| 0.7 | 0.80 | 1.42 | 1.27 |
| 0.5 | 1.29 | 1.99 | 1.80 |

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

## With examples in hand (added 2026-09-30)

Bruce's point: with no examples nothing can check the description, so
hedging is forced and the floor is obvious. The floor generalises to $n$
examples in hand (paper, Proposition 3): for every $n$,

$$p\,R_{p=1}(n) + (1-p)\,R^{\mathrm{ex}}(n) \le R^{\mathrm{desc}}(n) \le
p\,R_{p=1}(n) + (1-p)\,R^{\mathrm{ex}}(n) + H(Z \mid D_n),$$

and the identification term $H(Z \mid D_n)$ starts at $H(p)$ and never
rises. The floor $(1-p)R^{\mathrm{ex}}(n)$ falls with $n$, so the worth of
a precise unreliable description, measured as further examples saved
(Definition 2), rises as examples verify it, while a reliable description's
worth stays near $c = 1/(r a_0)$. Once examples have identified the
description and $R^{\mathrm{ex}}(n) \approx d/(2n)$,
$\mathrm{ESS}_n \approx n p c / (n + (1-p) c) \to p\,c$: worth $p$ times a
reliable description (90 against 100 at $r = 0.001$, $a_0 = 10$,
$p = 0.9$), and linear in $n$ while $n \ll (1-p)c$. The earlier statement
$\mathrm{ESS}_n \approx n p/(1-p)$ took $r \to 0$ before $n \to \infty$ and
was wrong as a limit.

**Numbers** ($d = 16$, $a_0 = 10$, `results/rq2-reliability/worth.csv`, mean
of three seeds; `worth.py`, `worth_figure.py`):

| $n$ in hand | 0 | 1 | 3 | 8 | 16 | 24 | 40 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $r = 0.001$, $p = 0.9$ | 23 | 30 | 32 | 30 | 37 | 57 | 74 |
| $r = 0.001$, $p = 1$ | 117 | 116 | 116 | 116 | 114 | 115 | |
| $r = 0.05$, $p = 0.9$ | 15 | 15 | 13 | 10 | 6 | 4 | 3 |
| $r = 0.05$, $p = 1$ | 17 | 16 | 15 | 11 | 7 | 4 | 3 |

`tests/test_propositions.py` checks the sandwich on every row. The
networks at $d = 5$, $r = 0.05$ show the same shape; the $p = 0.9$ network
under-values its description alone and meets the exact learner by about
five examples.
