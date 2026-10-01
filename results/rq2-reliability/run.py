"""RQ2: worth of an unreliable description. Writes three tables next to
this script:

  ess_map.csv     ESS over reliability p and precision r, by horizon,
                  for a learner with calibrated trust (q = p)
  trust.csv       regret when the learner's trust q differs from p
  tipping.csv     examples needed before the learner abandons a wrong
                  description (posterior weight on it falls below 1/2)

Settings: sigma_y = 1, base prior SNR a0, r = a_desc / a0.
"""

import csv
import pathlib

import numpy as np

from description_icl import gaussian as g
from description_icl import mixture as mx

HERE = pathlib.Path(__file__).parent
SETTINGS = ((4, 10.0), (16, 10.0), (16, 100.0), (64, 10.0))  # (d, a0)
PS = (1.0, 0.99, 0.9)
RS = (0.1, 0.01, 0.001)
HORIZONS = (1, 16, 200)
SEEDS = (0, 1, 2)


def write(name, rows):
    with (HERE / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def samples(d, K):
    return int(np.clip(1e9 / (K * d * d), 2_000, 50_000))


def base_G(d, a0, seed):
    K = int(2 / (min(RS) * a0) + 4 * d) + 8 + max(HORIZONS)
    X = g.gaussian_designs(d, K, samples(d, K), seed)
    return g.G_path(g.info_increments(X, a0)[0])[0]


def both_cases(d, a0, r, K, seed):
    """Paths given a correct and given a wrong description. Regret at any
    reliability p is the p-weighted average of the two."""
    S = samples(d, K)
    return (mx.simulate(d, a0, r * a0, 1.0, K, S, seed),
            mx.simulate(d, a0, r * a0, 0.0, K, S, seed + 1000))


def regret_at(cases, p, q, N):
    """(mean, standard error) regret of the description alone over horizon N."""
    (m1, s1), (m0, s0) = (mx.regret(c, q, N, with_se=True) for c in cases)
    return p * m1[0] + (1 - p) * m0[0], np.hypot(p * s1[0], (1 - p) * s0[0])


def ess_map():
    rows = []
    for d, a0 in SETTINGS:
        G0 = [base_G(d, a0, 100 + s) for s in SEEDS]
        for r in RS:
            cases = [both_cases(d, a0, r, max(HORIZONS), s) for s in SEEDS]
            for N in HORIZONS:
                for p in PS:
                    reg = [regret_at(c, p, p, N)[0] for c in cases]
                    ess = [g.first_crossing(g.regret_examples(G, N), x)[1]
                           for G, x in zip(G0, reg)]
                    rows.append(
                        dict(d=d, a0=a0, N=N, r=r, p=p,
                             regret=round(np.mean(reg), 4),
                             ess=round(np.mean(ess), 3),
                             ess_sd=round(np.std(ess, ddof=1), 3))
                    )
        print(f"ess_map d={d} a0={a0} done", flush=True)
    write("ess_map.csv", rows)


def trust():
    qs = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 1.0)
    rows = []
    d, a0 = 16, 10.0
    for r in (0.5, 0.05, 0.001):
        cases = both_cases(d, a0, r, 200, 0)
        for N in (1, 200):
            for p in (0.5, 0.7, 0.9):
                for q in qs:
                    reg, se = regret_at(cases, p, q, N)
                    rows.append(
                        dict(d=d, a0=a0, N=N, p=p, r=r, q=q,
                             regret=round(reg, 4), se=round(se, 4),
                             kl_p_q=round(_kl(p, q), 4))
                    )
    write("trust.csv", rows)


def _kl(p, q):
    if q in (0.0, 1.0):
        return float("inf")
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def tipping():
    rows = []
    K = 80
    for d, a0 in SETTINGS:
        for r in RS:
            # p = 0: every description is wrong
            paths = mx.simulate(d, a0, r * a0, 0.0, K, samples(d, K), 0)
            for q in (0.5, 0.9, 0.99, 0.999999):
                w_desc = 1 - np.exp(mx.log_weight_true(paths, q))
                med = np.median(w_desc, axis=0)
                q95 = np.quantile(w_desc, 0.95, axis=0)
                rows.append(
                    dict(d=d, a0=a0, r=r, q=q,
                         n_median=_first_below(med), n_95pct=_first_below(q95))
                )
        print(f"tipping d={d} a0={a0} done", flush=True)
    write("tipping.csv", rows)


def _first_below(curve, level=0.5):
    hit = np.nonzero(curve < level)[0]
    return int(hit[0]) if hit.size else -1


if __name__ == "__main__":
    ess_map()
    trust()
    tipping()
