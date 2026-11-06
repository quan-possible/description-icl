"""partial_sq.csv and partial_table.tex (the appendix table): the Bayes-optimal ESS of a reliable description that informs only k of the d directions, with n
demonstrations in hand; d = 16, a0 = 10 (sigma_y = 1).

Such a description leaves the fraction r of the prior variance in k directions and all of it in the other d - k, so
its prior precision is Lambda = diag(c I_k, I_{d-k} / a0) with c = 1 / (r a0), and its error is E tr(Lambda + W_n)^{-1}.
W_n = X_n^T X_n is rotation invariant, so which k directions does not matter. B batches of S prompts share the
demonstrations of the no-description curve; `se` is the standard error of the ESS across batches.

    python3 docs/paper/figs/partial_sq.py
"""

import csv
import pathlib

import numpy as np

here = pathlib.Path(__file__).parent
D, A0, K, S, B = 16, 10.0, 600, 2000, 8
KS = (1, 4, 8, 12, 15, 16)
RS = (0.1, 0.01, 0.001)
NS = (0, 1, 2, 4, 8, 10, 12, 14, 16, 20, 30, 50, 100, 200, 400)


def errors(seed):
    """R_demo(n) and, for each (k, r), R_desc(n), n = 0..K-1, on one batch of S prompts."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((S, K, D))
    prec = {None: np.full(D, 1 / A0)} | {(k, r): np.r_[np.full(k, 1 / (r * A0)), np.full(D - k, 1 / A0)] for k in KS for r in RS}
    out = {}
    for key, lam in prec.items():
        Sig = np.broadcast_to(np.diag(1 / lam), (S, D, D)).copy()
        tr = np.empty(K)
        for n in range(K):  # Sherman-Morrison: the posterior covariance after each demonstration
            tr[n] = np.trace(Sig, axis1=1, axis2=2).mean()
            u = np.einsum("sij,sj->si", Sig, X[:, n])
            Sig -= u[:, :, None] * u[:, None, :] / (1 + np.einsum("si,si->s", X[:, n], u))[:, None, None]
        out[key] = tr
    return out


def ess(R_demo, target, n):
    """n* - n, with n* the linearly interpolated root of R_demo(n*) = target (nan beyond the grid)."""
    below = np.nonzero(R_demo <= target)[0]
    if below.size == 0:
        return np.nan
    k = int(below[0])
    return (0.0 if k == 0 else (k - 1) + (R_demo[k - 1] - target) / (R_demo[k - 1] - R_demo[k])) - n


def table(rows, ks=(1, 4, 8, 12, 16), tab_ns=(0, 10, 100, 400)):
    """partial_table.tex: the ESS at a few n for loose and specific descriptions of k directions, with the limits."""
    get = lambda k, r, n: next(x["ess"] for x in rows if x["k"] == k and x["r"] == r and x["n"] == n)
    fmt = lambda v: f"{v + 1e-9:.1f}" if v < 100 else f"{v:.0f}"  # round half up (49.95 -> 50.0)
    lines = [r"\begin{tabular}{r" + "c" * (2 * len(tab_ns) + 2) + "}", r"\toprule",
             rf"& \multicolumn{{{len(tab_ns) + 1}}}{{c}}{{$c = 1$}} & \multicolumn{{{len(tab_ns) + 1}}}{{c}}{{$c = 100$}} \\",
             rf"\cmidrule(lr){{2-{len(tab_ns) + 2}}} \cmidrule(lr){{{len(tab_ns) + 3}-{2 * len(tab_ns) + 3}}}",
             "$k$ & " + " & ".join(2 * ([f"$n = {n}$" for n in tab_ns] + [r"$n \to \infty$"])) + r" \\", r"\midrule"]
    for k in ks:
        cells = []
        for r in (0.1, 0.001):
            cells += [fmt(get(k, r, n)) for n in tab_ns] + [fmt((1 - r) * k / (r * A0 * D))]
        lines.append(f"{k} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (here / "partial_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    runs = [errors(seed) for seed in range(B)]
    R_demo = np.mean([x[None] for x in runs], 0)
    rows = []
    for k in KS:
        for r in RS:
            R = np.mean([x[k, r] for x in runs], 0)
            assert abs(R[0] - A0 * (D - k + k * r)) < 1e-9 * A0 * D  # no demonstrations: the trace of the prior covariance
            c = 1 / (r * A0)
            for n in NS:
                per = [ess(x[None], x[k, r][n], n) for x in runs]
                rows.append(dict(k=k, r=r, c=round(c, 3), n=n, ess=round(ess(R_demo, R[n], n), 3), se=round(float(np.std(per, ddof=1) / np.sqrt(B)), 3),
                                 limit_n0=round((1 - r) * k, 2) if k < D else None, limit_many=round((1 - r) * c * k / D, 3)))
        print(f"k={k} done", flush=True)
    with (here / "partial_sq.csv").open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    table(rows)
