"""fallback_sensitivity.py: how the Bayes-optimal weight on the stated value, with a stated 50% (and 10%) chance of
being wrong, depends on what the reference assumes about the mean when the description is wrong ("the mean is
unknown" in the prompt). Fallbacks: the uniform over 100..999 used in the paper, the baseline prior N(500, 100^2) the
tasks are drawn from, and wider ones. First 200 tasks of the main run, c = 36 (sd 10), weight from the same
regression as the paper (stated value, mean of the numbers shown, constant). Prints one row per fallback."""
import numpy as np

import llm_test as T

T.S = 2000
t = T.make_tasks(T.MAIN)
K = 200
noise, tol = T.MAIN, 10.0
m = T.stated(t, (T.MAIN, 10, True, 0.5))[:K].astype(float)


def gauss(y, m0, v0):
    """posterior mean and log marginal likelihood of y (K, n) under the prior N(m0, v0)"""
    n = y.shape[1]
    d = y - np.reshape(m0, (-1, 1))
    mu = (noise**2 / v0 * m0 + y.sum(1)) / (noise**2 / v0 + n)
    ll = -0.5 * (np.log1p(n * v0 / noise**2) + (d**2).sum(1) / noise**2 - v0 * d.sum(1) ** 2 / (noise**2 * (noise**2 + n * v0))) - 0.5 * n * np.log(2 * np.pi * noise**2)
    return mu, ll


def uniform(y, lo, hi):
    """posterior mean and log marginal likelihood under the uniform prior on [lo, hi] (numerically, on a grid)"""
    g = np.linspace(lo, hi, 4001)
    lik = -0.5 * ((y[:, :, None] - g[None, None, :]) ** 2).sum(1) / noise**2 - 0.5 * y.shape[1] * np.log(2 * np.pi * noise**2)
    mx = lik.max(1, keepdims=True)
    w = np.exp(lik - mx)
    mu = (w * g).sum(1) / w.sum(1)
    ll = mx[:, 0] + np.log(w.mean(1))  # mean over the grid = integral / (hi - lo)
    return mu, ll


FALLBACKS = {"uniform 100-999 (paper)": lambda y: uniform(y, 100, 999), "N(500, 100^2) (baseline prior)": lambda y: gauss(y, np.full(len(y), 500.0), 100.0**2),
             "N(500, 200^2)": lambda y: gauss(y, np.full(len(y), 500.0), 200.0**2), "uniform 0-2000": lambda y: uniform(y, 0, 2000)}
NS = (1, 2, 4, 8, 16)
for q in (0.5, 0.9):
    print(f"\nstated chance of being wrong {round(100 * (1 - q))}%: Bayes-optimal weight on the stated value at n = {NS}")
    for name, fb in FALLBACKS.items():
        ws = []
        for n in NS:
            y = t["y"][:K, :n].astype(float)
            mu0, l0 = fb(y)
            mu1, l1 = gauss(y, m, tol**2)
            pi = 1 / (1 + (1 - q) / q * np.exp(np.clip(l0 - l1, -700, 700)))
            pred = pi * mu1 + (1 - pi) * mu0
            X = np.column_stack([m, y.mean(1), np.ones(K)])
            ws.append(np.linalg.lstsq(X, pred, rcond=None)[0][0])
        fixed = [q * 36 / (36 + n) for n in NS]
        print(f"  {name:32s} " + "  ".join(f"{w:.2f}" for w in ws))
    print(f"  {'fixed rule p*36/(36+n)':32s} " + "  ".join(f"{w:.2f}" for w in fixed))
