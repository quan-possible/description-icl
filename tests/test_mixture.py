import numpy as np
from scipy import stats

from descriptor_icl import gaussian as g
from descriptor_icl import mixture as mx


def test_log_ml_matches_joint_gaussian():
    d, a_base, a_desc, K, S = 3, 2.0, 0.3, 6, 4
    rng = np.random.default_rng(0)
    X = rng.standard_normal((S, K, d))
    Y = rng.standard_normal((S, K))
    m = rng.standard_normal((S, d))
    L, _, _ = mx._component(X, Y, m, a_desc)
    for s in range(S):
        cov = np.eye(K) + a_desc * X[s] @ X[s].T
        assert np.isclose(
            L[s, K], stats.multivariate_normal.logpdf(Y[s], X[s] @ m[s], cov)
        )


def test_reliable_description_reduces_to_gaussian():
    # p = q = 1: regret is the Gaussian description regret G_{a_desc}
    d, a_base, a_desc = 4, 5.0, 1.0
    paths = mx.simulate(d, a_base, a_desc, p=1.0, K=30, S=20_000, seed=1)
    G, se = g.G_path(paths.inc1)
    for N in (1, 10):
        r, r_se = mx.regret(paths, 1.0, N, with_se=True)
        assert abs(r[0] - G[N]) < 1e-9  # identical by construction
    # and the raw oracle-minus-learner estimate agrees with it
    X = g.gaussian_designs(d, 30, 20_000, seed=2)
    G2, se2 = g.G_path(g.info_increments(X, a_desc)[0])
    assert abs(G[10] - G2[10]) < 4 * np.hypot(se[10], se2[10])


def test_ignoring_description_gives_base_regret():
    d, a_base, a_desc = 4, 5.0, 1.0
    paths = mx.simulate(d, a_base, a_desc, p=0.6, K=30, S=200_000, seed=3)
    r, se = mx.regret(paths, 0.0, 5, with_se=True)
    X = g.gaussian_designs(d, 30, 50_000, seed=4)
    G, _ = g.G_path(g.info_increments(X, a_base)[0])
    base = g.regret_examples(G, 5)
    assert np.all(np.abs(r - base) < 5 * se + 0.01)


def test_identification_cost_limit_is_cross_entropy():
    # long horizon, well-separated components: regret excess over the
    # Gaussian term tends to the cross-entropy H(p, q)
    d, a_base, a_desc, p, q = 4, 20.0, 0.2, 0.7, 0.4
    paths = mx.simulate(d, a_base, a_desc, p, K=200, S=50_000, seed=5)
    lw = mx.log_weight_true(paths, q)
    excess = (lw[:, -1] - lw[:, 0]).mean()
    H = -(p * np.log(q) + (1 - p) * np.log(1 - q))
    assert abs(excess - H) < 0.01


def test_calibrated_trust_is_optimal():
    d, a_base, a_desc, p = 4, 5.0, 0.5, 0.7
    paths = mx.simulate(d, a_base, a_desc, p, K=10, S=200_000, seed=6)
    r = {q: mx.regret(paths, q, 1)[0] for q in (0.3, 0.5, 0.7, 0.9, 0.99)}
    assert min(r, key=r.get) == 0.7


def test_mean_prediction_limits():
    """Full trust predicts from the description, none from the base prior,
    and with no observations the weight is the trust itself."""
    paths = mx.simulate(3, 10.0, 0.5, 0.8, 4, 200, 0)
    assert np.allclose(mx.mean_prediction(paths, 1.0), paths.mu1)
    assert np.allclose(mx.mean_prediction(paths, 0.0), paths.mu0)
    first = mx.mean_prediction(paths, 0.8)[:, 0]
    assert np.allclose(first, 0.8 * paths.mu1[:, 0] + 0.2 * paths.mu0[:, 0])
