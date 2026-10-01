# Networks

**Status:** design fixed (log loss, decision 19 in
[experiments.md](experiments.md)). No model trained to completion. Design record: [experiments.md](experiments.md), section 6.

**Design.** Prefix embedding of Huang & Ge (2025): an optional descriptor
token carrying $m$, example tokens, a query token, two indicator columns. No
positional encoding. Architecture and optimisation of Garg et al. (2022):
12 layers, 8 heads, width 256, Adam at $10^{-4}$, fresh prompts every step.
$d = 5$, $a_0 = 10$, up to 15 examples, half the prompts without a
descriptor. $r$ and $p$ fixed per model and never stated.

**Loss.** Log loss with a predicted mean and spread (Genewein et al.),
decided by Bruce on 2026-09-30. One question per prompt, after a random
number of examples.

**Order.** One factor at a time: reliable precise descriptor; the same model
evaluated at $p = 0.9$; coarse descriptor; then training at $p = 0.9$. One seed per
configuration (decision 20).

**Targets** (`results/rq3-meta-trained/targets.csv`, squared error; log-loss
values in parentheses):

| $r$ | $p = 1$ | $p = 0.9$ |
| --- | --- | --- |
| 0.5 | 2.7 (2.2) | 2.1 (1.8) |
| 0.05 | 7.4 (6.7) | 4.4 (5.0) |

**Evidence so far.** An earlier pilot with the offset layout, a 6-layer
model, and log loss reached within 0.03 to 0.05 nats of the Bayes-optimal
predictor after 10,000 steps (ESS 6.68 against 6.55). It used a design since
replaced and is not a result.

**Compute.** One old-layout model trained at 54 steps/s on a Colab L4 and 24
on a T4 at batch 256; 18 models in parallel gave no more total throughput.
The Garg-size model is about eight times larger.
