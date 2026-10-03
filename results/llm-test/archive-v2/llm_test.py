"""The language-model test of the paper's Section 7: tasks, prompts, the ideal
learner, and the estimators. numpy only; score_model.py and analyze.py import it.

A task is a hidden mean w ~ N(500, 100^2) with K numbers w + N(0, 30^2), all
rounded to integers and kept inside 100..999. A prompt is the numbers, one per
line, preceded in the description conditions by one line that states a value m
and a tolerance: valid (w | m ~ N(m, tol^2)), invalid (the m of an independent
task), or exact (m = w). Every condition uses the same means and numbers.

    python3 results/llm-test/llm_test.py      # self-check on the ideal learner, ~1 minute
"""

import numpy as np

MEAN, S0, SIG, K = 500.0, 100.0, 30.0, 200
LO, HI = 100, 999
GRID = np.arange(LO, HI + 1)
TOLS = (30, 10, 3)
EPS = SIG**2 / S0**2  # 1 / a0
WINDOW = range(5, 11)  # readings in hand over which the ESS is averaged

WORDING = {
    "main": "Mean: {m} ± {tol}",
    "sentence": "The mean is about {m}, give or take {tol}.",
    "interval": "Mean in [{lo}, {hi}]",  # m - 2 tol to m + 2 tol
    "exact": "Mean: {m}",
}
# name: (which stated value, tolerance, wording); None is the prompt with no sentence
CONDITIONS = {
    "none": None,
    "valid30": ("valid30", 30, "main"),
    "valid10": ("valid10", 10, "main"),
    "valid3": ("valid3", 3, "main"),
    "exact": ("exact", 0, "exact"),
    "invalid30": ("invalid30", 30, "main"),
    "invalid10": ("invalid10", 10, "main"),
    "invalid3": ("invalid3", 3, "main"),
    **{f"{kind}{tol}_{wording}": (f"{kind}{tol}", tol, wording) for wording in ("sentence", "interval") for kind in ("valid", "invalid") for tol in TOLS},
}


def make_tasks(S, seed=0):
    """S tasks: w (S,), integer readings y (S, K), and the stated value m[name] (S,)."""
    rng = np.random.default_rng(seed)
    kept, have = [], 0
    while have < S:
        w = rng.normal(MEAN, S0, S)
        cols = {"y": np.rint(w[:, None] + SIG * rng.standard_normal((S, K))), "exact": np.rint(w)}
        for tol in TOLS:
            r = tol**2 / S0**2  # m | w, so that w | m ~ N(m, tol^2) and w ~ N(MEAN, S0^2)
            cols[f"valid{tol}"] = np.rint(MEAN + (1 - r) * (w - MEAN) + np.sqrt(r * (1 - r)) * S0 * rng.standard_normal(S))
            cols[f"invalid{tol}"] = np.rint(MEAN + np.sqrt(1 - r) * S0 * rng.standard_normal(S))
        ok = np.all([(v >= LO).reshape(S, -1).all(1) & (v <= HI).reshape(S, -1).all(1) for v in cols.values()], 0)
        for tol in TOLS:  # the interval wording stays three-digit
            for kind in ("valid", "invalid"):
                ok &= (cols[f"{kind}{tol}"] >= LO + 2 * tol) & (cols[f"{kind}{tol}"] <= HI - 2 * tol)
        kept.append({"w": w[ok], **{k: v[ok] for k, v in cols.items()}})
        have += int(ok.sum())
    cat = {k: np.concatenate([p[k] for p in kept])[:S] for k in kept[0]}
    return {"w": cat["w"], "y": cat["y"].astype(int), "m": {k: v.astype(int) for k, v in cat.items() if k not in ("w", "y")}}


def prompt(tasks, cond, i):
    head = ""
    if CONDITIONS[cond]:
        key, tol, wording = CONDITIONS[cond]
        m = tasks["m"][key][i]
        head = WORDING[wording].format(m=m, tol=tol, lo=m - 2 * tol, hi=m + 2 * tol) + "\n"
    return head + "".join(f"{v}\n" for v in tasks["y"][i])


# ---------- the ideal learner (known noise, Gaussian prior on the hidden value) ----------

def gaussian_path(y, m0, v0):
    """Predicting reading n + 1 from the first n, for n = 0..K-1, under the prior N(m0, v0):
    predictive mean and variance (S, K), and the log marginal likelihood of the first n readings."""
    prec, n = SIG**2 / v0, np.arange(K)
    before = np.concatenate([np.zeros((len(y), 1)), np.cumsum(y, 1)[:, :-1]], 1)
    mu = (prec * np.reshape(m0, (-1, 1)) + before) / (prec + n)
    pv = SIG**2 * (1 + 1 / (prec + n)) + 0 * mu
    logp = -0.5 * (np.log(2 * np.pi * pv) + (y - mu) ** 2 / pv)
    return mu, pv, np.concatenate([np.zeros((len(y), 1)), np.cumsum(logp, 1)[:, :-1]], 1)


def gaussian_regret(w, mu, pv):
    """Expected log loss of N(mu, pv) under the true reading law N(w, SIG^2), minus the oracle's."""
    return 0.5 * np.log(pv / SIG**2) + (SIG**2 + (w[:, None] - mu) ** 2) / (2 * pv) - 0.5


def erf(x):
    """Error function, Abramowitz and Stegun 7.1.26 (absolute error below 1.5e-7), on arrays."""
    t = 1 / (1 + 0.3275911 * np.abs(x))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return np.sign(x) * (1 - poly * np.exp(-x * x))


def gaussian_pmf(mu, var):
    """N(mu, var) rounded to integers, as a distribution over GRID (last axis)."""
    cdf = 0.5 * (1 + erf((np.append(GRID - 0.5, HI + 0.5) - np.asarray(mu)[..., None]) / np.sqrt(2 * np.asarray(var)[..., None])))
    p = np.diff(cdf, axis=-1)
    return p / p.sum(-1, keepdims=True)


def summarize(q, w, y):
    """From predictive distributions q (S, K, len(GRID)) over the next reading: the regret
    (expected log loss under the true reading distribution minus the oracle's), the log
    probability of the observed reading, and the distribution's mean and standard deviation."""
    p = gaussian_pmf(w, SIG**2)[:, None, :]
    logq = np.log(np.maximum(q, 1e-300))
    regret = (p * (np.log(np.maximum(p, 1e-300)) - logq)).sum(-1)
    obs = np.take_along_axis(logq, (y - LO)[..., None], -1)[..., 0]
    mean = (q * GRID).sum(-1)
    return regret, obs, mean, np.sqrt(np.maximum((q * GRID**2).sum(-1) - mean**2, 0))


# ---------- estimators ----------

def decreasing_fit(y):
    """Isotonic (non-increasing) least-squares fit, pool-adjacent-violators."""
    blocks = []
    for v in map(float, y):
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            (a, na), (b, nb) = blocks[-2], blocks[-1]
            blocks[-2:] = [[(a * na + b * nb) / (na + nb), na + nb]]
    return np.concatenate([[v] * n for v, n in blocks])


def ess(R_none, R_desc, window=WINDOW):
    """Definition 1 with the learner's own two mean regret curves, averaged over the window:
    n* - n, where the (monotone) no-sentence curve at n* equals the with-sentence curve at n.
    nan where the no-sentence curve never falls that low within the prompt, and -n where the
    with-sentence regret at n is no better than the no-sentence regret at 0 (the paper's ESS_n <= -n)."""
    R = decreasing_fit(R_none)
    out = []
    for n in window:
        below = np.nonzero(R <= R_desc[n])[0]
        if below.size == 0:
            return np.nan
        k = int(below[0])
        out.append((0.0 if k == 0 else (k - 1) + (R[k - 1] - R_desc[n]) / max(R[k - 1] - R[k], 1e-12)) - n)
    return float(np.mean(out))


def weights(mean, m, y, ns=WINDOW):
    """For each n in ns (numbers in hand, n >= 1), the least-squares fit over tasks of
    mean_n = a_n m + b_n ybar_n + const: the weight a_n on the stated value and b_n on the mean of
    the numbers. The constant absorbs any pull of the predictions toward a fixed value. The ideal
    learner has a_n = c / (c + n) and b_n = n / (c + n). Returns a, b, one entry per n."""
    out = np.empty((2, len(ns)))
    for j, n in enumerate(ns):
        X = np.column_stack([m, y[:, :n].mean(1), np.ones(len(m))])
        out[:, j] = np.linalg.lstsq(X, mean[:, n], rcond=None)[0][:2]
    return out[0], out[1]


def implied_c(mean, m, y, window=WINDOW):
    """The strength, in numbers, that the weights correspond to: n a_n / b_n, averaged over the window."""
    a, b = weights(mean, m, y, window)
    return float(np.mean(np.array(window) * a / b))


def hedged_mean(tasks, c, p):
    """Mean prediction of the Bayes-optimal predictor that gives the line of condition c prior
    probability p of being valid: the two Gaussian learners' means, weighted by the posterior
    probability that the line is valid given the numbers so far."""
    key, tol, _ = CONDITIONS[c]
    mu1, _, L1 = gaussian_path(tasks["y"], tasks["m"][key], max(tol, 1e-3) ** 2)
    mu0, _, L0 = gaussian_path(tasks["y"], MEAN, S0**2)
    pi = 1 / (1 + (1 - p) / p * np.exp(np.clip(L0 - L1, -700, 700)))
    return pi * mu1 + (1 - pi) * mu0


def demo():
    """The estimators recover the ideal learner: ESS = c - 1/a0 and implied c = c."""
    t = make_tasks(400, seed=1)
    assert prompt(t, "none", 0).count("\n") == K and prompt(t, "valid10", 0).startswith("Mean: ") and prompt(t, "valid10_interval", 0).startswith("Mean in [")
    mu0, pv0, _ = gaussian_path(t["y"], MEAN, S0**2)
    R_none = summarize(gaussian_pmf(mu0, pv0), t["w"], t["y"])[0].mean(0)
    assert abs(R_none[5] - gaussian_regret(t["w"], mu0, pv0).mean(0)[5]) < 2e-3  # pmf route = closed form
    for tol in TOLS:
        c = SIG**2 / tol**2
        mu, pv, _ = gaussian_path(t["y"], t["m"][f"valid{tol}"], tol**2)
        R, _, mean, _ = summarize(gaussian_pmf(mu, pv), t["w"], t["y"])
        got, got_c = ess(R_none, R.mean(0)), implied_c(mean, t["m"][f"valid{tol}"].astype(float), t["y"])
        print(f"tolerance {tol:2d}: ESS {got:6.2f} (ideal {c - EPS:6.2f}); implied c {got_c:6.2f} (ideal {c:6.2f})")
        assert abs(got - (c - EPS)) < 0.35 * c + 0.3 and abs(got_c - c) < 0.02 * c + 0.05
        hedge = implied_c(hedged_mean(t, f"invalid{tol}", 0.9), t["m"][f"invalid{tol}"].astype(float), t["y"])
        print(f"              a hedging learner's strength for an invalid line: {hedge:.2f}")
        assert abs(hedge) < 0.3
    print("ok")


if __name__ == "__main__":
    demo()
