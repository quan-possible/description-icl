"""Write single_gaussian.csv: on the p = 0.9 and p = 0.99 prompts of the network setting
(d = 5, a0 = 10), the one-step regret with a description and n examples of
the Bayes-optimal mixture predictive and of the best single Gaussian, the one
matching the mixture's mean and variance, which is all a mean-and-variance
output head can represent; plus each one's ESS alone (crossing with the
examples-only curve of targets.csv) and with n examples in hand.

    uv run python results/rq3-meta-trained/single_gaussian.py
"""

import csv
import pathlib

import numpy as np

from description_icl import gaussian as g

here = pathlib.Path(__file__).parent
D, A0, K, S = 5, 10.0, 21, 100_000
R_ex = np.array([float(x["regret"]) for x in csv.DictReader((here / "targets.csv").open()) if x["r"] == "none"])
ORACLE = 0.5 * np.log(2 * np.pi * np.e)


def ridge_pred(X, Y, mu, a, n, xq):
    """Predictive mean and variance at xq after n examples under the prior N(mu, a I)."""
    Xn, Yn = X[:, :n, :], Y[:, :n]
    Sig = np.linalg.inv(np.eye(D)[None] / a + np.einsum("ski,skj->sij", Xn, Xn))
    mean = np.einsum("sij,sj->si", Sig, mu / a + np.einsum("ski,sk->si", Xn, Yn))
    return np.einsum("si,si->s", xq, mean), 1 + np.einsum("si,sij,sj->s", xq, Sig, xq)


def nll(y, mu, v):
    return 0.5 * np.log(2 * np.pi * v) + 0.5 * (y - mu) ** 2 / v


rows = []
for P, r in [(p, r) for p in (0.9, 0.99) for r in (0.01, 0.1)]:
    rng = np.random.default_rng(300)
    X = rng.standard_normal((S, K, D))
    m = np.sqrt(A0 - r * A0) * rng.standard_normal((S, D))
    correct = rng.random(S) < P
    w = np.where(correct[:, None], m + np.sqrt(r * A0) * rng.standard_normal((S, D)), np.sqrt(A0) * rng.standard_normal((S, D)))
    Y = np.einsum("skd,sd->sk", X, w) + rng.standard_normal((S, K))
    logml1 = np.zeros(S); logml0 = np.zeros(S)  # running log marginal likelihoods of the two components
    curves = {"mixture": [], "single": []}
    for n in range(K):
        xq, y = X[:, n, :], Y[:, n]
        m1, v1 = ridge_pred(X, Y, m, r * A0, n, xq)
        m0, v0 = ridge_pred(X, Y, np.zeros_like(m), A0, n, xq)
        pi = 1 / (1 + np.exp(np.clip(-(np.log(P) + logml1 - np.log(1 - P) - logml0), -500, 500)))
        mix = -np.logaddexp(np.log(pi) - nll(y, m1, v1), np.log1p(-pi) - nll(y, m0, v0))
        mu_g = pi * m1 + (1 - pi) * m0
        v_g = pi * v1 + (1 - pi) * v0 + pi * (1 - pi) * (m1 - m0) ** 2
        curves["mixture"].append(mix.mean() - ORACLE); curves["single"].append(nll(y, mu_g, v_g).mean() - ORACLE)
        logml1 -= nll(y, m1, v1); logml0 -= nll(y, m0, v0)  # the query becomes example n + 1
    for who, c in curves.items():
        for n in range(K):
            e = float(g.first_crossing(R_ex, c[n])[1]) - n
            rows.append(dict(r=r, p=P, who=who, n=n, regret=round(c[n], 4), ess_n="" if np.isnan(e) else round(e, 2)))
    print(f"p={P} r={r}: alone mixture {curves['mixture'][0]:.3f} single {curves['single'][0]:.3f}; ESS alone",
          {w: [x["ess_n"] for x in rows if x["r"] == r and x["p"] == P and x["who"] == w and x["n"] == 0][0] for w in curves}, flush=True)
with (here / "single_gaussian.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
