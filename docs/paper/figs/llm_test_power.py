"""How many tasks the proposed language-model test (Section 7) needs.

Simulates the ideal learner on the test's own design: a hidden value
w ~ N(500, 100^2), readings w + N(0, 30^2), and a relevant description with
tolerance 30, 10 or 3. One pass over a task scores reading k from the k - 1
before it. The ESS follows Definition 1 from the two mean regret curves (the
baseline curve made monotone by isotonic regression), averaged over n = 5..10
readings in hand. The ideal value is c - 1/a0 with c = (30 / tolerance)^2.

Two ways of scoring are compared: the log loss of the one observed reading, and
the expected log loss under the true reading distribution, which is available
when the model's whole predictive distribution is read at each position.
Prints the mean and standard deviation of the ESS estimate over replications.

    python3 docs/paper/figs/llm_test_power.py     # about 5 minutes
"""

import numpy as np

S0, SIG, K, REPS = 100.0, 30.0, 200, 100
EPS = SIG**2 / S0**2  # 1 / a0: the base prior's precision in units of the noise


def decreasing_fit(y):
    """Isotonic (non-increasing) least-squares fit, pool-adjacent-violators."""
    blocks = []
    for v in map(float, y):
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            (a, na), (b, nb) = blocks[-2], blocks[-1]
            blocks[-2:] = [[(a * na + b * nb) / (na + nb), na + nb]]
    return np.concatenate([[v] * n for v, n in blocks])


def regret_curves(S, tol, rng, whole_distribution):
    """Mean regret at n = 0..K-1 without and with the description (units of the noise)."""
    c = SIG**2 / tol**2
    m = rng.normal(0, np.sqrt(S0**2 - tol**2), S) / SIG
    w = m + rng.normal(0, tol / SIG, S)
    e = rng.standard_normal((S, K))
    y = w[:, None] + e
    before = np.concatenate([np.zeros((S, 1)), np.cumsum(y, 1)[:, :-1]], 1)  # sum of earlier readings
    n = np.arange(K)
    curves = []
    for prec, mean0 in ((EPS, 0.0), (c, m[:, None])):
        mu, v = (prec * mean0 + before) / (prec + n), 1 + 1 / (prec + n)
        if whole_distribution:
            excess = (1 + (w[:, None] - mu) ** 2) / (2 * v) - 0.5
        else:
            excess = (y - mu) ** 2 / (2 * v) - e**2 / 2
        curves.append((0.5 * np.log(v) + excess).mean(0))
    return curves


def ess(R_ex, target, n):
    R_ex = decreasing_fit(R_ex)
    below = np.nonzero(R_ex <= target)[0]
    if below.size == 0:
        return np.nan
    k = int(below[0])
    return (0.0 if k == 0 else (k - 1) + (R_ex[k - 1] - target) / max(R_ex[k - 1] - R_ex[k], 1e-12)) - n


def demo():
    rng = np.random.default_rng(0)
    for whole in (False, True):
        print("regret from", "the whole predictive distribution" if whole else "the observed reading")
        for tol in (30, 10, 3):
            for S in (500, 2000, 8000):
                est = []
                for _ in range(REPS):
                    R_ex, R_desc = regret_curves(S, tol, rng, whole)
                    est.append(np.nanmean([ess(R_ex, R_desc[n], n) for n in range(5, 11)]))
                print(f"  tolerance {tol:2d}  ideal {SIG**2 / tol**2 - EPS:6.2f}  S = {S:4d}  ESS {np.nanmean(est):7.2f} +- {np.nanstd(est):5.2f}")


if __name__ == "__main__":
    demo()
