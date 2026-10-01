"""Exact Bayes-optimal regret and ESS for reliable descriptions (p = 1).

Setting: y = w.x + eps, x ~ N(0, I_d), eps ~ N(0, sigma_y^2). Under a prior
w ~ N(m, s^2 I) the posterior covariance does not depend on m or on the
observed y, so every regret below is an expectation over the inputs X alone.
Only the ratio a = s^2 / sigma_y^2 (prior signal-to-noise) matters.

Log-loss regret against the oracle that knows w obeys the chain rule

    R_N = 1/2 E[log det Sigma_0 - log det Sigma_N],

so with G_a(k) = 1/2 E log det(I + a X_k^T X_k), X_k a k-by-d Gaussian matrix,

    description alone, horizon N:   G_{a_desc}(N)
    n examples, horizon N:          G_{a_base}(n + N) - G_{a_base}(n)

All regrets are in nats. Squared-error regret is in units of sigma_y^2.
"""

import numpy as np
from scipy import integrate, special, stats


def info_increments(X, a):
    """Per-step information gains and posterior traces along nested designs.

    X has shape (S, K, d). With M_k = I + a X_k^T X_k (first k rows), returns
    inc (S, K) with inc[:, k-1] = log det M_k - log det M_{k-1}, and
    tr (S, K+1) with tr[:, k] = trace(M_k^{-1}).
    """
    S, K, d = X.shape
    Minv = np.broadcast_to(np.eye(d), (S, d, d)).copy()
    inc = np.empty((S, K))
    tr = np.empty((S, K + 1))
    tr[:, 0] = d
    for k in range(K):
        x = X[:, k, :]
        u = np.einsum("sij,sj->si", Minv, x)
        denom = 1.0 + a * np.einsum("si,si->s", x, u)
        inc[:, k] = np.log(denom)
        Minv -= (a / denom)[:, None, None] * u[:, :, None] * u[:, None, :]
        tr[:, k + 1] = np.einsum("sii->s", Minv)
    return inc, tr


def gaussian_designs(d, K, S, seed):
    return np.random.default_rng(seed).standard_normal((S, K, d))


def G_path(inc):
    """G_a(k) for k = 0..K (mean over samples) and its standard error."""
    cum = 0.5 * np.concatenate(
        [np.zeros((inc.shape[0], 1)), np.cumsum(inc, axis=1)], axis=1
    )
    return cum.mean(axis=0), cum.std(axis=0, ddof=1) / np.sqrt(cum.shape[0])


def G_exact_d1(a, k):
    """G_a(k) in d = 1 by quadrature: 1/2 E log(1 + a chi2_k)."""
    if k == 0:
        return 0.0
    f = lambda t: 0.5 * np.log1p(a * t) * stats.chi2.pdf(t, k)
    # split at the mode region so quad resolves the k = 1 singularity at 0
    val = integrate.quad(f, 0, k, limit=200)[0]
    val += integrate.quad(f, k, np.inf, limit=200)[0]
    return val


def expected_logdet_wishart(d, k):
    """E log det(X_k^T X_k) for k >= d (Bartlett)."""
    i = np.arange(1, d + 1)
    return d * np.log(2.0) + special.digamma((k - i + 1) / 2.0).sum()


def first_crossing(curve, target):
    """Smallest n with curve[n] <= target, and a linearly interpolated version.

    Returns (n_int, n_cont); both are nan if the curve never reaches target.
    """
    below = np.nonzero(curve <= target)[0]
    if below.size == 0:
        return np.nan, np.nan
    n = int(below[0])
    if n == 0:
        return 0, 0.0
    hi, lo = curve[n - 1], curve[n]
    return n, (n - 1) + (hi - target) / (hi - lo)


def regret_examples(G_base, N):
    """Log-loss regret over horizon N after n examples, for n = 0..K-N."""
    return G_base[N:] - G_base[: len(G_base) - N]


def ess_logloss(G_base, G_desc, N):
    """ESS at horizon N: examples needed to match the description's regret."""
    return first_crossing(regret_examples(G_base, N), G_desc[N])


def ess_long_horizon(G_base, d, a_base, a_desc):
    """N -> infinity limit: examples whose expected information gain reaches
    the description's, (d/2) log(a_base / a_desc)."""
    return first_crossing(-G_base, -0.5 * d * np.log(a_base / a_desc))


def ess_squared(tr_base_mean, d, a_base, a_desc):
    """Single-query ESS under squared error: a_base E tr(M_n^{-1}) <= d a_desc."""
    return first_crossing(a_base * tr_base_mean, d * a_desc)


def ess_balanced(a_base, a_desc):
    """Balanced designs (X^T X = n I): the posterior after n examples is
    isotropic with 1/a = 1/a_base + n, so the ESS is horizon-free."""
    return 1.0 / a_desc - 1.0 / a_base


# Proportional limit: d, n -> infinity with n / d = gamma and total SNR
# rho = d * a_base fixed. X^T X then follows the Marchenko-Pastur law and the
# per-dimension trace and log-determinant have closed forms (Tulino and
# Verdu 2004, eqs. 2.120-2.121 style eta and Shannon transforms).


def _mp_F(x, z):
    return (
        np.sqrt(x * (1 + np.sqrt(z)) ** 2 + 1) - np.sqrt(x * (1 - np.sqrt(z)) ** 2 + 1)
    ) ** 2


def mp_trace(gamma, rho):
    """lim (1/d) tr (I + a X^T X)^{-1}, n = gamma d, a = rho / d."""
    beta, snr = 1.0 / gamma, rho * gamma
    return 1.0 - _mp_F(snr, beta) / (4 * beta * snr)


def mp_logdet(gamma, rho):
    """lim (1/d) log det(I + a X^T X), n = gamma d, a = rho / d."""
    beta, snr = 1.0 / gamma, rho * gamma
    F = _mp_F(snr, beta)
    return (
        np.log(1 + snr - F / 4)
        + np.log(1 + snr * beta - F / 4) / beta
        - F / (4 * beta * snr)
    )


def ess_proportional(rho, r):
    """Per-dimension ESS (single-query, long-horizon) in the proportional
    limit for a description leaving a fraction r of the prior variance."""
    from scipy.optimize import brentq

    hi = 10.0 / (r * rho) + 10.0
    single = brentq(lambda t: mp_trace(t, rho) - r, 1e-9, hi)
    long = brentq(lambda t: mp_logdet(t, rho) - np.log(1 / r), 1e-9, hi)
    return single, long
