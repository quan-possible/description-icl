"""ess_grid.py: does the paper's headline contrast depend on its illustration values (d = 16, a0 = 10)?
Bayes-optimal zero-demonstration ESS of a c = 100 description at p = 1 and p = 0.9, over other dimensions d and
baseline variances a0. R_desc(0) = [1 - p^2 (1 - r)] a0 d (Proposition 2, r = 1 / (c a0)); R_demo(n) =
E tr(I / a0 + X'X)^{-1} by Monte Carlo (4,000 draws of the n x d Gaussian inputs per n); ESS_0 is the first crossing
of the baseline curve, interpolated linearly between integers (Definition 1). Limits for comparison: c + d + 1 at
p = 1 (Proposition 1(ii)) and the ceiling p^2 d (Proposition 2) as a0 grows."""
import numpy as np

rng = np.random.default_rng(0)


def baseline(d, a0, nmax, draws=4000):
    out = [a0 * d]
    for n in range(1, nmax + 1):
        X = rng.standard_normal((draws, n, d))
        ev = np.linalg.eigvalsh(np.einsum("sni,snj->sij", X, X)) if n >= d else np.concatenate(
            [np.linalg.eigvalsh(np.einsum("sin,sjn->sij", X, X)), np.zeros((draws, d - n))], 1)
        out.append(np.mean(np.sum(1 / (1 / a0 + np.maximum(ev, 0)), 1)))
    return np.array(out)


def ess0(curve, target):
    k = int(np.argmax(curve <= target))
    return k - 1 + (curve[k - 1] - target) / (curve[k - 1] - curve[k])


if __name__ == "__main__":
    c = 100
    print("  d    a0   ESS_0 p=1   limit c+d+1   ESS_0 p=0.9   ceiling p^2 d")
    for d in (5, 16, 50):
        for a0 in (1, 10, 100):
            curve = baseline(d, a0, c + d + 12, draws=4000 if d < 50 else 1500)
            r = 1 / (c * a0)
            e1, e9 = (ess0(curve, (1 - p * p * (1 - r)) * a0 * d) for p in (1.0, 0.9))
            print(f"{d:3d} {a0:5d} {e1:10.1f} {c + d + 1:12d} {e9:12.1f} {0.81 * d:13.2f}")
