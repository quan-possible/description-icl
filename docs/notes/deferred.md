# Deferred material

Everything cut from the simplest version of the paper on 2026-09-30, with
where it lives. Nothing here is lost; it is out of the first paper until the
simple results exist.

| Material | State | Where |
| --- | --- | --- |
| The prediction horizon: ESS falls with $N$, its limit is the information-matching sample size, and the ordering reverses for precise unreliable descriptions | Computed on the RQ1 and RQ2 grids; written up | [archive/paper-with-horizon.tex](../paper/archive/paper-with-horizon.tex); `results/rq1-single-query-gap/`, `results/rq2-reliability/` |
| Asymptotic ESS of an unreliable description, $p\,\tfrac d2\log(1/r) - H(p)$ | Derived informally, checked at one point | Same draft, section 5 |
| Boundary of the horizon reversal | Not derived | |
| Proportional-limit closed forms (Marchenko–Pastur) | Derived, matches simulation within 2% at $d = 64$ | `gaussian.ess_proportional`; RQ1 README |
| Cost of misspecified reliability, $\mathrm{KL}(p\,\|\,q)$ | Derived and validated | `results/rq2-reliability/trust.csv`; RQ2 README |
| Speed of identifying a wrong description (1 to 3 examples) | Computed | `results/rq2-reliability/tipping.csv` |
| Effective-reliability readout for networks | Code exists | `results/rq3-meta-trained/evaluate.py` (trust output) |
| Comparison with language-model crossover curves | Literature only | [2026-09-29-neurips-assessment.md](2026-09-29-neurips-assessment.md), "Link to LLMs" |
| LSTM architecture | Removed from code; in Git history before `bfcf2c8` | |
| Huang & Ge's input-mean descriptor as a second condition | Idea only | `docs/DESIGN.md`, Deferred |
| Squared-error output as a check | Idea only | `docs/DESIGN.md`, Deferred |

The assessment's view was that the horizon results are the most novel part
of the project. They return as one section once the simple paper's three
results exist.
