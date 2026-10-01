# Setting and ESS definition

**Task.** $w \in \mathbb{R}^d$; examples $x \sim \mathcal{N}(0, I_d)$,
$y = w^\top x + \epsilon$, $\epsilon \sim \mathcal{N}(0, 1)$. Base prior
$w \sim \mathcal{N}(0, a_0 I_d)$. Follows Garg et al. (2022).

**Description.** A vector $m$ and a precision ratio $r$: if correct,
$w \sim \mathcal{N}(m, r a_0 I_d)$. Drawn as
$m \sim \mathcal{N}(0, (1-r) a_0 I_d)$ so the marginal law of $w$ is the base
prior with or without a description. Correct with probability $p$; otherwise
$w$ is a fresh draw from the base prior. Bayes-optimal prior: the robust
mixture $p\,\mathcal{N}(m, r a_0 I) + (1-p)\,\mathcal{N}(0, a_0 I)$
(Schmidli et al. 2014).

**Regret.** Cumulative log loss over $N$ sequential predictions, each true
label revealed before the next, minus the oracle's that knows $w$. In nats.
Follows Genewein et al. (2025). $N = 1$ is the one-step (single-query) case.

**ESS.** The $n$ at which the Bayes-optimal predictor with $n$ examples and no
description has the same regret as with the description and no examples,
interpolated between integers. Both are the same predictor with different
conditioning information. Prediction-based ESS was left open by Reimherr et
al. (2021).

**Code.** `src/description_icl/gaussian.py` ($p = 1$),
`src/description_icl/mixture.py` ($p < 1$). Validated against a direct
log-determinant, $d = 1$ quadrature, the Wishart limit, balanced designs, the
long-horizon limit, and the joint Gaussian likelihood (`tests/`).
