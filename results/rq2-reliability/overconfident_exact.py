"""Closed-form one-step regrets with no demonstrations (n = 0), for the
numbers quoted in the paper's overconfidence paragraph: the overconfident
predictor (reliance q = 1) given a description of reliability p, and the
predictor with no description. Writes overconfident_exact.csv next to this
script.

With sigma_y = 1, s_0^2 = a_0 and A = |x|^2 ~ chi^2_d, the q = 1 predictive is
N(m^T x, v) with v = 1 + r a_0 A. Its regret given Z is
    Z = 1:  (1/2) E log v
    Z = 0:  (1/2) E [ log v + ((2 - r) a_0 A + 1) / v - 1 ]
because (w - m)^T x has variance r a_0 A when Z = 1 and (2 - r) a_0 A when
Z = 0. No description: (1/2) E log(1 + a_0 A). All are one-dimensional
integrals over A, done by quadrature.

    uv run python results/rq2-reliability/overconfident_exact.py
"""

import csv
import pathlib

import numpy as np
from scipy import integrate, stats

D, A0, P = 16, 10.0, 0.9


def ex(f, d=D):
    pdf = stats.chi2(d).pdf
    return integrate.quad(lambda a: f(a) * pdf(a), 0, np.inf, limit=200)[0]


def overconfident(r, p=P, d=D, a0=A0):
    v = lambda a: 1 + r * a0 * a
    r1 = 0.5 * ex(lambda a: np.log(v(a)), d)
    r0 = 0.5 * ex(lambda a: np.log(v(a)) + ((2 - r) * a0 * a + 1) / v(a) - 1, d)
    return p * r1 + (1 - p) * r0


none = 0.5 * ex(lambda a: np.log(1 + A0 * a))
rows = [dict(d=D, a0=A0, p=P, r=r, q=1, regret=round(overconfident(r), 3)) for r in (0.1, 0.01, 0.001)]
rows.append(dict(d=D, a0=A0, p="", r="none", q="", regret=round(none, 3)))
with (pathlib.Path(__file__).parent / "overconfident_exact.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(*rows, sep="\n")

if __name__ == "__main__":  # check against a direct simulation of the q = 1 predictor at d = 16
    rng = np.random.default_rng(0)
    S, r = 400_000, 0.01
    m = rng.normal(0, np.sqrt((1 - r) * A0), (S, D))
    z = rng.random(S) < P
    w = np.where(z[:, None], m + rng.normal(0, np.sqrt(r * A0), (S, D)), rng.normal(0, np.sqrt(A0), (S, D)))
    x = rng.normal(0, 1, (S, D))
    y = (w * x).sum(1) + rng.normal(0, 1, S)
    v = 1 + r * A0 * (x ** 2).sum(1)
    mu = (m * x).sum(1)
    oracle = 0.5 * np.log(2 * np.pi) + 0.5 * (y - (w * x).sum(1)) ** 2
    reg = 0.5 * np.log(2 * np.pi * v) + (y - mu) ** 2 / (2 * v) - oracle
    mc, se = reg.mean(), reg.std() / np.sqrt(S)
    assert abs(mc - overconfident(r)) < 4 * se + 1e-3, (mc, se, overconfident(r))
    print("simulation check", round(mc, 3), "+/-", round(se, 3), "closed form", round(overconfident(r), 3))
