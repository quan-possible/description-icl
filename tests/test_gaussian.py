import numpy as np

from description_icl import gaussian as g


def test_increments_match_direct_logdet():
    X = g.gaussian_designs(d=5, K=12, S=3, seed=0)
    a = 0.7
    inc, tr = g.info_increments(X, a)
    for k in (1, 4, 12):
        M = np.eye(5) + a * np.einsum("ski,skj->sij", X[:, :k], X[:, :k])
        assert np.allclose(inc[:, :k].sum(1), np.linalg.slogdet(M)[1])
        assert np.allclose(tr[:, k], np.trace(np.linalg.inv(M), axis1=1, axis2=2))


def test_d1_matches_quadrature():
    a = 4.0
    X = g.gaussian_designs(d=1, K=20, S=200_000, seed=1)
    G, se = g.G_path(g.info_increments(X, a)[0])
    for k in (1, 2, 5, 20):
        assert abs(G[k] - g.G_exact_d1(a, k)) < 4 * se[k]


def test_large_snr_matches_wishart_logdet():
    d, k, a = 4, 10, 1e8
    X = g.gaussian_designs(d, k, S=100_000, seed=2)
    G, se = g.G_path(g.info_increments(X, a)[0])
    exact = 0.5 * (d * np.log(a) + g.expected_logdet_wishart(d, k))
    assert abs(G[k] - exact) < 4 * se[k] + 1e-6


def test_balanced_design_has_no_horizon_gap():
    # n = ESS balanced examples leave an isotropic posterior equal to the
    # description's prior, so the regret paths from there are identical.
    d, a_base, a_desc = 4, 2.0, 0.1
    n = g.ess_balanced(a_base, a_desc)
    assert n == 9.5
    a_desc = 1.0 / (1.0 / a_base + 8)  # pick a description worth n = 8 = 2d
    ctx = np.sqrt(d) * np.tile(np.eye(d), (2, 1))  # X^T X = 8 I
    Xq = g.gaussian_designs(d, 30, S=50, seed=3)
    X = np.concatenate([np.broadcast_to(ctx, (50, 8, d)), Xq], axis=1)
    inc_ex, _ = g.info_increments(X, a_base)
    inc_desc, _ = g.info_increments(Xq, a_desc)
    assert np.allclose(inc_ex[:, 8:], inc_desc)


def test_ess_converges_to_long_horizon_limit():
    d, a_base, a_desc = 1, 4.0, 0.25
    G_base = np.array([g.G_exact_d1(a_base, k) for k in range(2300)])
    G_desc = np.array([g.G_exact_d1(a_desc, k) for k in range(2001)])
    limit = g.ess_long_horizon(G_base, d, a_base, a_desc)[1]
    e1 = g.ess_logloss(G_base, G_desc, 1)[1]
    e2000 = g.ess_logloss(G_base, G_desc, 2000)[1]
    assert e1 > limit
    assert abs(e2000 - limit) < 0.05 * limit


def test_proportional_limit_matches_finite_d():
    d, gamma, rho = 200, 1.5, 50.0
    X = g.gaussian_designs(d, int(gamma * d), S=20, seed=5)
    inc, tr = g.info_increments(X, rho / d)
    assert abs(tr[:, -1].mean() / d - g.mp_trace(gamma, rho)) < 0.01
    assert abs(inc.sum(1).mean() / d - g.mp_logdet(gamma, rho)) < 0.01
    single, long = g.ess_proportional(rho, r=0.2)
    assert single > long
