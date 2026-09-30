# Horizon dependence of the ESS

**Status:** computed on the RQ1 grid; the limit and the inequality are
classical; the magnitude is ours. Written up in
[../paper/archive/paper-with-horizon.tex](../paper/archive/paper-with-horizon.tex),
not in the current paper.

**Result.** For a reliable description the ESS decreases with the horizon
$N$. Its limit is the information-matching sample size: the $n^\star$ with
$G_{a_0}(n^\star) = \tfrac d2 \log(1/r)$ (Clarke & Barron 1990: cumulative
log-loss regret equals mutual information). Its one-step value is larger:
$\mathrm{ESS}_1 \ge \mathrm{ESS}_\infty$, with equality under balanced
designs and near-equality in $d = 1$.

**Why.** One-step regret is governed by $\operatorname{tr}\Sigma$
(A-optimality), information by $\log\det\Sigma$ (D-optimality; Chaloner &
Verdinelli 1995). A description contracts every eigenvalue by $r$; $n < d$
examples contract $n$ eigenvalues to near zero and leave the rest. By AM–GM,
isotropic contraction gives the smaller trace at equal log-determinant.

**Numbers** ($a_0 = 10$):

| $d$ | $r$ | $\mathrm{ESS}_1$ | $\mathrm{ESS}_\infty$ | ratio |
| --- | --- | --- | --- | --- |
| 1 | 0.5 | 0.50 | 0.40 | 1.26 |
| 16 | 0.5 | 7.62 | 2.23 | 3.42 |
| 64 | 0.5 | 31.6 | 6.93 | 4.56 |
| 64 | 0.05 | 62.1 | 31.1 | 2.00 |

The ratio runs from 1.01 to 8.5 over the grid, largest at $d = 64$,
$a_0 = 100$, $r = 0.9$; between 1.01 and 1.35 in $d = 1$; exactly 1 under
balanced designs. The proposal's "up to three times" is not a bound.

**Under squared error** the horizon dependence is also present ($d = 5$,
$r = 0.05$: 7.4 at $N = 1$, 5.9 at $N = 200$) but the limit has no closed
form.

**Real-world reading.** $N = 1$ is a normal prompt; large $N$ is a session
with feedback. A description is worth more for one answer than its
information content, by a factor that grows with the task's dimension.
