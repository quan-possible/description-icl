"""RQ1: ESS of a reliable description (p = 1) across dimension, SNR,
precision, and horizon. Writes ess.csv next to this script.

a0 = s_0^2 / sigma_y^2 is the base prior's signal-to-noise ratio and
r = s_l^2 / s_0^2 the fraction of prior variance the description leaves.
"""

import csv
import pathlib

import numpy as np

from description_icl import gaussian as g

DIMS = (1, 2, 4, 8, 16, 32, 64)
A0S = (1.0, 10.0, 100.0)
RS = (0.1, 0.01, 0.001)
HORIZONS = (1, 4, 16, 64, 256)
SEEDS = (0, 1, 2)
BUDGET = 4e9  # rough flop budget per design batch, sets the sample count


def ess_table(d, a0, seed):
    n_max = int(2 / (min(RS) * a0) + 4 * d) + 8  # covers the largest ESS
    K = n_max + max(HORIZONS)
    S = int(np.clip(BUDGET / (K * d * d), 500, 50_000))
    X = g.gaussian_designs(d, K, S, seed)
    inc0, tr0 = g.info_increments(X, a0)
    G0, _ = g.G_path(inc0)
    out = {}
    for r in RS:
        Gd, _ = g.G_path(g.info_increments(X[:, : max(HORIZONS)], r * a0)[0])
        row = {f"ess_N{N}": g.ess_logloss(G0, Gd, N)[1] for N in HORIZONS}
        row["ess_inf"] = g.ess_long_horizon(G0, d, a0, r * a0)[1]
        row["ess_sq"] = g.ess_squared(tr0.mean(0), d, a0, r * a0)[1]
        out[r] = row
    return out


def main():
    rows = []
    for d in DIMS:
        for a0 in A0S:
            tables = [ess_table(d, a0, seed) for seed in SEEDS]
            for r in RS:
                row = {"d": d, "a0": a0, "r": r}
                for key in tables[0][r]:
                    vals = np.array([t[r][key] for t in tables])
                    row[key] = round(vals.mean(), 3)
                    row[key + "_sd"] = round(vals.std(ddof=1), 3)
                row["ratio_1_inf"] = round(row["ess_N1"] / row["ess_inf"], 3)
                # high-SNR, coarse-description approximation (n < d)
                row["ratio_approx"] = round(
                    (1 - r) * np.log1p(d * a0) / np.log(1 / r), 3
                )
                row["balanced_ess"] = round(g.ess_balanced(a0, r * a0), 3)
                rows.append(row)
            print(f"d={d} a0={a0} done", flush=True)
    path = pathlib.Path(__file__).with_name("ess.csv")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
