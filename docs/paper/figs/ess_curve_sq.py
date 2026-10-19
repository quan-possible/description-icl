"""ess_curve_sq.csv, the data behind ess_sq.pdf (Figure 2) and quoted ESS values:
the Bayes-optimal ESS of a description with n = 0, 1, 10, 100 demonstrations in hand, for
p = 1, 0.99, 0.9 over a log-spaced grid of r; d = 16, a0 = 10 (sigma_y = 1).

The error is the squared distance of the prediction from the noiseless label, E[(yhat - w^T x)^2]. Averaged over
the query x ~ N(0, I) it is exact given the prompt's demonstrations: the posterior variance
tr(Sigma) of w for one Gaussian component, and for the mixture (law of total variance)

    pi tr(Sigma_1) + (1 - pi) tr(Sigma_0) + pi (1 - pi) |mu_1 - mu_0|^2,

with pi the posterior probability that the description is relevant. With no demonstrations
the error is closed form (Proposition 2), so the n = 0 rows, the r -> 0 limit
(r = 0 rows) and the bound for a predictor told whether the description is relevant (r = -1
rows) need only the demonstrations-only curve. B batches of S prompts; `se` is the standard
error of the ESS across batches.

    python3 docs/paper/figs/ess_curve_sq.py
"""

import csv
import pathlib

import numpy as np

here = pathlib.Path(__file__).parent
D, A0, K, S, B = 16, 10.0, 260, 4000, 8
RS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
PS = (1.0, 0.99, 0.9)
NS = (0, 1, 10, 100)


def r0(r, p):
    """Bayes-optimal error with the description and no demonstrations (Proposition 2)."""
    return A0 * D * (p * r + (1 - p) * (1 + p * (1 - r)))


def regrets(r, seed):
    """R_demo(n) and R_desc(n) for each p, n = 0..K-1, on one batch of prompts."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((S, K, D))
    noise = rng.standard_normal((S, K))
    m = np.sqrt(A0 * (1 - r)) * rng.standard_normal((S, D))
    w = {1: m + np.sqrt(A0 * r) * rng.standard_normal((S, D)), 0: np.sqrt(A0) * rng.standard_normal((S, D))}
    Y = {z: np.einsum("skd,sd->sk", X, w[z]) + noise for z in w}
    Sig = {1: np.broadcast_to(r * A0 * np.eye(D), (S, D, D)).copy(), 0: np.broadcast_to(A0 * np.eye(D), (S, D, D)).copy()}
    mu = {(c, z): (m if c == 1 else np.zeros((S, D))).copy() for c in (0, 1) for z in w}
    L = {(c, z): np.zeros(S) for c in (0, 1) for z in w}  # log marginal likelihood of the demonstrations
    tr = {c: np.empty(K) for c in (0, 1)}
    R = {p: np.empty(K) for p in PS if p < 1}
    for n in range(K):
        t = {c: np.trace(Sig[c], axis1=1, axis2=2) for c in (0, 1)}
        for c in (0, 1):
            tr[c][n] = t[c].mean()
        for p in R:  # E over Z: relevant with probability p
            tot = 0.0
            for z, pz in ((1, p), (0, 1 - p)):
                pi = 1 / (1 + np.exp(np.log1p(-p) - np.log(p) + L[0, z] - L[1, z]))
                gap = ((mu[1, z] - mu[0, z]) ** 2).sum(1)
                tot += pz * (pi * t[1] + (1 - pi) * t[0] + pi * (1 - pi) * gap).mean()
            R[p][n] = tot
        x = X[:, n]
        for c in (0, 1):
            u = np.einsum("sij,sj->si", Sig[c], x)
            var = 1 + np.einsum("si,si->s", x, u)
            for z in w:
                err = Y[z][:, n] - np.einsum("si,si->s", x, mu[c, z])
                L[c, z] += -0.5 * (np.log(2 * np.pi * var) + err**2 / var)
                mu[c, z] += u * (err / var)[:, None]
            Sig[c] -= u[:, :, None] * u[:, None, :] / var[:, None, None]
    return tr[0], {1.0: tr[1]} | R


def ess(R_demo, target, n):
    """n* - n, with n* the linearly interpolated root of R_demo(n*) = target."""
    below = np.nonzero(R_demo <= target)[0]
    if below.size == 0:
        return np.nan
    k = int(below[0])
    return (0.0 if k == 0 else (k - 1) + (R_demo[k - 1] - target) / (R_demo[k - 1] - R_demo[k])) - n


rows = []


def add(p, r, n, per_batch, pooled):
    rows.append(dict(p=p, r=r, n=n, worth=round(pooled, 3), se=round(float(np.std(per_batch, ddof=1) / np.sqrt(B)), 3)))


for r in RS:
    runs = [regrets(r, seed) for seed in range(B)]
    R_demo = np.mean([x[0] for x in runs], 0)
    if r == RS[0]:
        for p in PS[1:]:  # n = 0 limits, from the closed form: r -> 0, and the bound for the predictor told Z
            add(p, 0, 0, [ess(x[0], r0(0, p), 0) for x in runs], ess(R_demo, r0(0, p), 0))
            add(p, -1, 0, [ess(x[0], (1 - p) * A0 * D, 0) for x in runs], ess(R_demo, (1 - p) * A0 * D, 0))
    for p in PS:
        R_desc = np.mean([x[1][p] for x in runs], 0)
        assert abs(R_desc[0] - r0(r, p)) < 0.05 * r0(r, p), (r, p, R_desc[0], r0(r, p))  # Monte Carlo agrees with the closed form
        for n in NS:
            add(p, r, n, [ess(x[0], x[1][p][n], n) for x in runs], ess(R_demo, R_desc[n], n))
    print(f"r={r} done", flush=True)

with (here / "ess_curve_sq.csv").open("w", newline="") as f:
    out = csv.DictWriter(f, fieldnames=list(rows[0]))
    out.writeheader()
    out.writerows(rows)
