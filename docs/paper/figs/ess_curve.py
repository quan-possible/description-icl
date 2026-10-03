"""ess_curve.csv, the data behind ess.pdf and the paper's quoted ESS values:
the ESS of a description with n = 0, 1, 10, 100 demonstrations in hand, for
p = 1, 0.99, 0.9 over a log-spaced grid of r; d = 16, a0 = 10 (sigma_y = 1).

Monte Carlo design. Each prompt has one fixed query, scored after every
number of demonstrations n, so each regret curve is monotone per prompt and
the with- and without-description curves share their inputs. The regret of
the Bayes-optimal (mixture) predictor is computed as in Proposition 2,

    R_desc(n) = p R_1(n) + (1 - p) R_ex(n) + identification term,

where R_1 and R_ex are exact given the inputs (half the mean log predictive
variance). The identification term is the KL divergence from the true case's
predictive to the mixture, integrated over the query's label on a grid of M
normal quantiles, separately for relevant and irrelevant descriptions, at the n the
figure needs. B batches of S prompts, seeds 0..B-1; `se` is the standard error
of the ESS across batches.

Rows with r = 0 hold the limit of the n = 0 ESS as r -> 0; rows with r = -1
hold the ESS at which R_ex(n) = (1 - p) R_ex(0), the bound for a predictor
told whether the description is relevant.

    python3 docs/paper/figs/ess_curve.py     # about an hour
"""

import csv
import pathlib
from statistics import NormalDist

import numpy as np

here = pathlib.Path(__file__).parent
D, A0, K, S, B, M = 16, 10.0, 260, 8000, 8, 256
RS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
PS = (1.0, 0.99, 0.9)
NS = (0, 1, 10, 100)
Q = np.array([NormalDist().inv_cdf((i + 0.5) / M) for i in range(M)])  # label quantiles


def log_normal(y, mean, var):
    return -0.5 * (np.log(2 * np.pi * var) + (y - mean) ** 2 / var)


def regrets(r, seed):
    """R_ex(n) and R_desc(n) for each p, n = 0..K-1, on one batch of prompts."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((S, K, D))  # drawn first: the same inputs for every r
    xq = rng.standard_normal((S, D))
    noise = rng.standard_normal((S, K))
    m = np.sqrt(A0 * (1 - r)) * rng.standard_normal((S, D))
    w = {True: m + np.sqrt(A0 * r) * rng.standard_normal((S, D)), False: np.sqrt(A0) * rng.standard_normal((S, D))}
    Y = {z: np.einsum("skd,sd->sk", X, w[z]) + noise for z in w}

    # two Gaussian components: the description's prior (1) and the base prior (0)
    Sig = {1: np.broadcast_to(r * A0 * np.eye(D), (S, D, D)).copy(), 0: np.broadcast_to(A0 * np.eye(D), (S, D, D)).copy()}
    mu = {(1, z): m.copy() for z in w} | {(0, z): np.zeros((S, D)) for z in w}
    L = {(c, z): np.zeros(S) for c in (0, 1) for z in w}  # log marginal likelihood of the demonstrations
    G = {c: np.empty(K) for c in (0, 1)}
    ident = {(p, z): np.zeros(K) for p in PS if p < 1 for z in w}  # filled at n in NS

    for n in range(K):
        v = {c: 1 + np.einsum("si,sij,sj->s", xq, Sig[c], xq) for c in (0, 1)}
        for c in (0, 1):
            G[c][n] = 0.5 * np.log(v[c]).mean()
        for z in w if n in NS else ():
            mean = {c: np.einsum("sd,sd->s", mu[c, z], xq) for c in (0, 1)}
            y = (mean[int(z)] + np.sqrt(v[int(z)]) * Q[:, None])  # (M, S): labels from the true case's predictive
            lf = {c: log_normal(y, mean[c], v[c]) for c in (0, 1)}
            for p in PS:
                if p < 1:
                    a1, a0 = np.log(p) + L[1, z], np.log1p(-p) + L[0, z]
                    log_f = np.logaddexp(a1 + lf[1], a0 + lf[0]) - np.logaddexp(a1, a0)
                    ident[p, z][n] = (lf[int(z)] - log_f).mean()
        x = X[:, n]
        for c in (0, 1):
            u = np.einsum("sij,sj->si", Sig[c], x)
            var = 1 + np.einsum("si,si->s", x, u)
            for z in w:
                err = Y[z][:, n] - np.einsum("si,si->s", x, mu[c, z])
                L[c, z] += log_normal(err, 0.0, var)
                mu[c, z] += u * (err / var)[:, None]
            Sig[c] -= u[:, :, None] * u[:, None, :] / var[:, None, None]

    R_desc = {p: G[1] if p == 1 else p * (G[1] + ident[p, True]) + (1 - p) * (G[0] + ident[p, False]) for p in PS}
    return G[0], R_desc


def ess(R_ex, target, n):
    """n* - n, with n* the linearly interpolated root of R_ex(n*) = target."""
    below = np.nonzero(R_ex <= target)[0]
    if below.size == 0:
        return np.nan
    k = int(below[0])
    return (0.0 if k == 0 else (k - 1) + (R_ex[k - 1] - target) / (R_ex[k - 1] - R_ex[k])) - n


rows = []


def add(p, r, n, per_batch, pooled):
    if np.isfinite(pooled):
        rows.append(dict(p=p, r=r, n=n, worth=round(pooled, 3), se=round(float(np.std(per_batch, ddof=1) / np.sqrt(B)), 3)))


for r in (1e-6,) + RS:
    runs = [regrets(r, seed) for seed in range(B)]
    R_ex = np.mean([x[0] for x in runs], 0)
    for p in PS:
        R_desc = np.mean([x[1][p] for x in runs], 0)
        if r == 1e-6:  # the limit as r -> 0, and the bound for the predictor told Z
            if p < 1:
                add(p, 0, 0, [ess(x[0], x[1][p][0], 0) for x in runs], ess(R_ex, R_desc[0], 0))
                add(p, -1, 0, [ess(x[0], (1 - p) * x[0][0], 0) for x in runs], ess(R_ex, (1 - p) * R_ex[0], 0))
            continue
        for n in NS:
            add(p, r, n, [ess(x[0], x[1][p][n], n) for x in runs], ess(R_ex, R_desc[n], n))
    print(f"r={r} done", flush=True)

with (here / "ess_curve.csv").open("w", newline="") as f:
    out = csv.DictWriter(f, fieldnames=list(rows[0]))
    out.writeheader()
    out.writerows(rows)
