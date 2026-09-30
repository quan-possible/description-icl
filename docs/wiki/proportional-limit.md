# Proportional limit

**Status:** derived; matches simulation within about 2% at $d = 64$ and
$d = 200$. Not in the current paper.

As $d, n \to \infty$ with $n/d = \gamma$ and $\rho = d\,a_0$ fixed, $X^\top X$
follows the Marchenko–Pastur law and (Tulino & Verdú 2004, eqs. 2.52, 2.61,
2.62):

- $\mathrm{ESS}_1 / d$ solves $\eta(\gamma) = r$, with $\eta$ the
  $\eta$-transform (the per-dimension trace);
- $\mathrm{ESS}_\infty / d$ solves $\mathcal{V}(\gamma) = \log(1/r)$, with
  $\mathcal{V}$ the Shannon transform (the per-dimension log-determinant).

Code: `gaussian.mp_trace`, `gaussian.mp_logdet`, `gaussian.ess_proportional`.
Closest precedent: Lu et al. (2025) use the same machinery for the risk of
linear attention. Useful as an appendix check; the grid answers the question
without it.
