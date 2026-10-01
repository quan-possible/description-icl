"""Numerical checks of the paper's two propositions (docs/paper/paper.tex).

Proposition 1: at high SNR the one-step ESS of a reliable description solves
psi((d-n)/2) - psi(d/2) = log r, and equals (d-1)(1-r) + O(1/(rd)).

Proposition 2: for every p and r, the one-step regret of an unreliable
description lies between p R_1(r) + (1-p) R_ex(0) and that plus H(p).
"""

import csv
import math
import pathlib

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma
from scipy.stats import chi2

from description_icl import gaussian as g

ROOT = pathlib.Path(__file__).resolve().parents[1]


def regret_no_examples(d, a):
    """1/2 E log(1 + a chi2_d) by quadrature: the one-step regret with prior
    SNR a and no examples."""
    z = np.linspace(1e-9, chi2.ppf(1 - 1e-12, d), 400_001)
    f = 0.5 * np.log1p(a * z) * chi2.pdf(z, d)
    return np.trapezoid(f, z)


def ess_digamma(d, r):
    """The n solving psi((d-n)/2) - psi(d/2) = log r."""
    return brentq(lambda n: digamma((d - n) / 2) - digamma(d / 2) - math.log(r), 0, d - 1e-9)


def test_precision_high_snr():
    """The exact ESS at a0 = 1e4 matches the digamma solution and the
    (d-1)(1-r) rule within its O(1/(rd)) error term."""
    a0 = 1e4
    for d, r in [(16, 0.5), (16, 0.2), (64, 0.5), (64, 0.2)]:
        K = d + 1
        X = g.gaussian_designs(d, K, 4000, seed=0)
        G0, _ = g.G_path(g.info_increments(X, a0)[0])
        Gd, _ = g.G_path(g.info_increments(X[:, :1], r * a0)[0])
        exact = g.ess_logloss(G0, Gd, 1)[1]
        assert abs(exact - ess_digamma(d, r)) < 0.15, (d, r, exact)
        assert abs(exact - (d - 1) * (1 - r)) < 0.15 + 2 / (r * d), (d, r, exact)


def test_reliability_sandwich():
    """Every one-step cell of the RQ2 table with p < 1 sits between the two
    bounds of Proposition 2 at n = 0, and R_ex(0) at d = 16, a0 = 10 is 2.51 nats."""
    rows = list(csv.DictReader((ROOT / "results/rq2-reliability/ess_map.csv").open()))
    rows = [{k: float(v) for k, v in row.items()} for row in rows if row["N"] == "1"]
    r1 = {(x["d"], x["a0"], x["r"]): x["regret"] for x in rows if x["p"] == 1.0}
    assert abs(regret_no_examples(16, 10.0) - 2.51) < 0.01
    checked = 0
    for x in rows:
        if x["p"] == 1.0:
            continue
        p = x["p"]
        lo = p * r1[(x["d"], x["a0"], x["r"])] + (1 - p) * regret_no_examples(int(x["d"]), x["a0"])
        hi = lo - p * math.log(p) - (1 - p) * math.log(1 - p)
        tol = 0.02  # Monte Carlo error of the table
        assert lo - tol <= x["regret"] <= hi + tol, (x, lo, hi)
        checked += 1
    assert checked == 4 * 2 * 3  # (d, a0) settings, p < 1, r


def test_reliability_with_examples_sandwich():
    """Proposition 2 with examples in hand on the computed worth table: with n examples in hand,
    p R_1(n) + (1-p) R_ex(n) <= R_desc(n) <= that + E[-log pi_n(Z)], and the
    identification term starts at H(p) and never rises."""
    rows = list(csv.DictReader((ROOT / "results/rq2-reliability/worth.csv").open()))
    rows = [{k: (float(v) if v != "" else float("nan")) for k, v in x.items()} for x in rows]
    p = 0.9
    H = -p * math.log(p) - (1 - p) * math.log(1 - p)
    checked = 0
    for seed in (0.0, 1.0, 2.0):
        for r in (0.1, 0.01, 0.001):
            one = {x["n"]: x for x in rows if x["seed"] == seed and x["r"] == r and x["p"] == 1.0}
            mix = {x["n"]: x for x in rows if x["seed"] == seed and x["r"] == r and x["p"] == p and x["q"] == p}
            assert abs(mix[0.0]["ident"] - H) < 0.03  # sample fraction of correct descriptions
            for n in range(101):
                lo = p * one[n]["R_desc"] + (1 - p) * mix[n]["R_ex"]
                hi = lo + mix[n]["ident"]
                tol = 0.03  # Monte Carlo, 2000 prompts, and R_1 from a separate draw
                assert lo - tol <= mix[n]["R_desc"] <= hi + tol, (seed, r, n, lo, mix[n]["R_desc"], hi)
                if n:
                    assert mix[n]["ident"] <= mix[n - 1.0]["ident"] + 0.01
                checked += 1
    assert checked == 3 * 3 * 101


def test_precision_additive_rule():
    """Proposition 1 on the full RQ1 grid: the additive rule (d-1)(1-r) + 1/(r a0)
    is within 2.2 examples of the exact one-step ESS at a0 >= 10, and the
    universal bound ESS <= 1/(r a0) + d + 2 holds in every cell."""
    rows = [{k: float(v) for k, v in x.items()} for x in csv.DictReader((ROOT / "results/rq1-single-query-gap/ess.csv").open())]
    assert len(rows) == 63
    for x in rows:
        c = 1 / (x["r"] * x["a0"])
        assert x["ess_N1"] <= c + x["d"] + 2 + 0.5, x  # 0.5 for Monte Carlo
        if x["a0"] >= 10:
            assert abs(x["ess_N1"] - ((x["d"] - 1) * (1 - x["r"]) + c)) <= 2.2, x
