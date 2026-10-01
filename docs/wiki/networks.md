# Networks

**Status:** four models trained (2026-09-30, one per setting, 40k steps on
Colab L4); results in the paper's section 6. One continuation run pending. Design record: [experiments.md](experiments.md), section 6.

**Design.** Prefix embedding of Huang & Ge (2025): an optional description
token carrying $m$, example tokens, a query token, two indicator columns. No
positional encoding. Architecture and optimisation of Garg et al. (2022):
12 layers, 8 heads, width 256, Adam at $10^{-4}$, fresh prompts every step.
$d = 5$, $a_0 = 10$, up to 15 examples, half the prompts without a
description. $r$ and $p$ fixed per model and never stated.

**Loss.** Log loss with a predicted mean and spread (Genewein et al.),
decided by Bruce on 2026-09-30. One question per prompt, after a random
number of examples.

**Order.** One factor at a time: reliable precise description; the same model
evaluated at $p = 0.9$; coarse description; then training at $p = 0.9$. One seed per
configuration (decision 20).

**Targets** (`results/rq3-meta-trained/targets.csv`, squared error; log-loss
values in parentheses):

| $r$ | $p = 1$ | $p = 0.9$ |
| --- | --- | --- |
| 0.5 | 2.7 (2.2) | 2.1 (1.8) |
| 0.05 | 7.4 (6.7) | 4.4 (5.0) |

**Results** (`results/rq3-meta-trained/summary.csv`; ESS network vs exact):

| $r$ | $p_{\text{train}}$ | $p_{\text{test}}$ | Network | Exact (trained trust) | Exact (calibrated) |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 1 | 1 | 6.7 | 6.7 | |
| 0.5 | 1 | 1 | 2.2 | 2.3 | |
| 0.05 | 0.9 | 0.9 | 3.9 | 5.0 | |
| 0.5 | 0.9 | 0.9 | 1.8 | 1.9 | |
| 0.05 | 1 | 0.9 | 0.7 | 0.0 | 5.0 |
| 0.5 | 1 | 0.9 | 1.6 | 1.7 | 1.9 |

Three of four matching-distribution models sit on the exact learner's regret
curve within 0.02 nats. The precise unreliable model under-values its
description (implied trust 0.85, regret 0.3 nats above Bayes from the
description alone); its training loss was still falling at 40k, and a
continuation to 80k steps is pending. Models trained on reliable
descriptions trust fully: on $p = 0.9$ prompts the precise one gets 0.7
examples' worth from a description that a calibrated learner would value
at 5.0.

**Pilot.** The gap to the exact learner fell from 0.07 nats at 5k steps to
0.02 at 20k and stayed there to 40k (`pilot.csv`). An earlier pilot with a
since-replaced design is not a result.

**Compute.** One old-layout model trained at 54 steps/s on a Colab L4 and 24
on a T4 at batch 256; 18 models in parallel gave no more total throughput.
The Garg-size model is about eight times larger.
