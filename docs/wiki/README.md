# Wiki

One page per result or topic, kept current whether or not the paper uses
it. The paper takes what it needs from here; nothing is "deferred".

| Page | What it holds |
| --- | --- |
| [Setting and ESS definition](setting.md) | The task, the description, regret, and the effective sample size |
| [Precision](precision.md) | ESS of a reliable description; the $d(1-r)$ rule |
| [Reliability](reliability.md) | The regret floor and the saturation of ESS in precision |
| [Horizon dependence](horizon.md) | ESS as a function of the prediction horizon; the single-query gap |
| [Horizon and reliability](horizon-reliability.md) | The asymptotic formula and the reversal |
| [Proportional limit](proportional-limit.md) | Closed forms from the Marchenko–Pastur law |
| [Misspecified reliability](misspecified-reliability.md) | Cost of assuming the wrong $p$ |
| [Identifying a wrong description](identification.md) | How many examples until a wrong description is abandoned |
| [Networks](networks.md) | The meta-training design, targets, and results |
| [Language models](llms.md) | What published LLM results say about the predictions |
| [Literature](literature.md) | Closest work and how each differs |

Design decisions live in [experiments.md](experiments.md); the current paper in
[../paper/](../paper/paper.tex); the venue assessment in
[../notes/2026-09-29-neurips-assessment.md](../notes/2026-09-29-neurips-assessment.md).

Notation throughout: $\sigma_y = 1$; $a_0 = s_0^2/\sigma_y^2$ is the base
prior signal-to-noise ratio; $r = s_\ell^2/s_0^2$ is the precision ratio; $p$
is the true reliability and $q$ the predictor's assumed reliability; $N$ is
the prediction horizon; $n$ the number of examples.
