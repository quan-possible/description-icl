"""The fully trusting predictor's squared error with n demonstrations in hand, quoted in the paper's
Section 5: the r = 0.001 description at p = 0.9, d = 16, a0 = 10, against the error without the description.
It predicts with the reliable description's posterior mean whatever the description's relevance.

    python3 docs/paper/figs/trust_sq.py
"""

import numpy as np

D, A0, R, P, K, S, B = 16, 10.0, 0.001, 0.9, 31, 4000, 8
c, eps = 1 / (R * A0), 1 / A0
trust, demo = np.zeros(K), np.zeros(K)
for seed in range(B):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((S, K, D)); noise = rng.standard_normal((S, K))
    m = np.sqrt(A0 * (1 - R)) * rng.standard_normal((S, D))
    w = {1: m + np.sqrt(A0 * R) * rng.standard_normal((S, D)), 0: np.sqrt(A0) * rng.standard_normal((S, D))}
    for n in range(K):
        Xn = X[:, :n]; W = np.einsum("ski,skj->sij", Xn, Xn)
        A1 = c * np.eye(D) + W
        demo[n] += np.trace(np.linalg.inv(eps * np.eye(D) + W), axis1=1, axis2=2).mean() / B
        for z, pz in ((1, P), (0, 1 - P)):
            y = np.einsum("ski,si->sk", Xn, w[z]) + noise[:, :n]
            mu = np.linalg.solve(A1, (c * m + np.einsum("ski,sk->si", Xn, y))[..., None])[..., 0]
            trust[n] += pz * ((mu - w[z]) ** 2).sum(1).mean() / B
print("closed form at n = 0:", A0 * D * (1 - (2 * P - 1) * (1 - R)), " limit as r -> 0:", 2 * (1 - P) * A0 * D)
print("first n with trust worse than no description:", int(np.argmax(trust > demo)))
for n in (0, 10, 14, 15, 20, 30):
    print(f"n={n}: trust {trust[n]:.1f}  no description {demo[n]:.1f}")
