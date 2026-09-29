"""Exact Bayes-optimal regret for unreliable descriptions (robust mixture).

Units: sigma_y = 1, so prior variances are the ratios a = s^2 / sigma_y^2.

Generative process. The description states a vector m ~ N(0, (a_base - a_desc) I).
With probability p it is correct, w ~ N(m, a_desc I); otherwise w ~ N(0, a_base I)
independently of m. Either way w is marginally N(0, a_base I).

A learner that trusts the description with probability q uses the prior
q N(m, a_desc I) + (1 - q) N(0, a_base I). Its marginal likelihood is
q ML_1 + (1 - q) ML_0, so its cumulative log-loss regret follows from the two
Gaussian components' sequential log marginal likelihoods. Writing c for the
true component and pi_c(k) for the learner's posterior weight on it after k
observations, the regret over observations n+1..n+N is

    E[ 1/2 sum of the true component's information gains ]
      + E[ log pi_c(n + N) - log pi_c(n) ],

a Gaussian term (independent of y) plus an identification term.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass
class Paths:
    correct: np.ndarray  # (S,) bool, whether the description was correct
    L1: np.ndarray  # (S, K+1) cumulative log ML, description component
    L0: np.ndarray  # (S, K+1) cumulative log ML, base component
    inc1: np.ndarray  # (S, K) log predictive-variance ratio, description component
    inc0: np.ndarray  # (S, K) same, base component


def simulate(d, a_base, a_desc, p, K, S, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((S, K, d))
    m = np.sqrt(a_base - a_desc) * rng.standard_normal((S, d))
    correct = rng.random(S) < p
    w = np.where(
        correct[:, None],
        m + np.sqrt(a_desc) * rng.standard_normal((S, d)),
        np.sqrt(a_base) * rng.standard_normal((S, d)),
    )
    Y = np.einsum("skd,sd->sk", X, w) + rng.standard_normal((S, K))
    L1, inc1 = _component(X, Y, m, a_desc)
    L0, inc0 = _component(X, Y, np.zeros((S, d)), a_base)
    return Paths(correct, L1, L0, inc1, inc0)


def _component(X, Y, mu0, a):
    """Sequential log marginal likelihood under the prior N(mu0, a I)."""
    S, K, d = X.shape
    mu = mu0.copy()
    Sig = np.broadcast_to(a * np.eye(d), (S, d, d)).copy()
    L = np.zeros((S, K + 1))
    inc = np.empty((S, K))
    for k in range(K):
        x = X[:, k, :]
        u = np.einsum("sij,sj->si", Sig, x)
        var = 1.0 + np.einsum("si,si->s", x, u)
        err = Y[:, k] - np.einsum("si,si->s", x, mu)
        L[:, k + 1] = L[:, k] - 0.5 * (np.log(2 * np.pi * var) + err**2 / var)
        inc[:, k] = np.log(var)
        mu += u * (err / var)[:, None]
        Sig -= u[:, :, None] * u[:, None, :] / var[:, None, None]
    return L, inc


def log_weight_true(paths, q):
    """log of the learner's posterior weight on the true component, (S, K+1).

    q = 1 or q = 0 give a learner committed to one component; its weight on
    the other is zero (log weight -inf), which the regret handles separately.
    """
    with np.errstate(divide="ignore"):
        l1 = np.log(q) + paths.L1
        l0 = np.log1p(-q) + paths.L0
    mix = logsumexp(np.stack([l1, l0]), axis=0)
    return np.where(paths.correct[:, None], l1, l0) - mix


def regret(paths, q, N, with_se=False):
    """Regret over horizon N of description + n examples, for n = 0..K-N.

    The learner trusts the description with probability q; the data come from
    the process that produced `paths` (true reliability p).
    """
    S, K = paths.inc1.shape
    inc_true = np.where(paths.correct[:, None], paths.inc1, paths.inc0)
    cum = 0.5 * np.concatenate([np.zeros((S, 1)), np.cumsum(inc_true, 1)], 1)
    if q in (0.0, 1.0):
        # committed learner: regret is oracle minus that component's log ML,
        # i.e. the true component's Gaussian term plus L_true - L_used
        L_true = np.where(paths.correct[:, None], paths.L1, paths.L0)
        L_used = paths.L1 if q == 1.0 else paths.L0
        total = cum + L_true - L_used
    else:
        total = cum + log_weight_true(paths, q)
    per_sample = total[:, N:] - total[:, : K + 1 - N]
    mean = per_sample.mean(0)
    if with_se:
        return mean, per_sample.std(0, ddof=1) / np.sqrt(S)
    return mean
