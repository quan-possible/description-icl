# RQ3: do meta-trained networks match the Bayes-optimal learner?

**What it shows.** Nothing yet. The pipeline runs end to end; no model has
been trained to completion.

## Design

The design, a worked example, and the reasons are in
[docs/DESIGN.md](../../docs/DESIGN.md), section 6, which is authoritative.

| Quantity | Value |
| --- | --- |
| $d$, `a0` | 5, 10; the model sees $w \sim \mathcal{N}(0, I)$ and noise variance 0.1 |
| Prompt | Optional description, 0 to 15 examples, one question |
| `r` | 0.5, 0.05 (one model each) |
| $p$ | 1 in stage 1, 0.9 in stage 2 (one model each) |
| Model | Transformer, 12 layers, 8 heads, width 256 (Garg et al.), no positions |
| Output | One number, the answer to the question |
| Loss | Squared error |
| Training | Adam, learning rate 1e-4, batch 1024; step count set by a pilot |
| Seeds | 0, 1, 2 |

## Run

```bash
uv run python results/rq3-meta-trained/targets.py   # the exact learner's targets
uv run python results/rq3-meta-trained/train.py --p 1.0 --r 0.05 --seed 0
uv run python results/rq3-meta-trained/evaluate.py tmp/rq3/p1.0_r0.05_d5_s0.pt
```

Checkpoints go to `tmp/rq3/` and stay out of Git. `evaluate.py --p-test`
evaluates at a reliability different from training.

## Targets

`targets.csv` holds the exact learner's regret and ESS under squared error
at this setting. The ESS targets are 2.7 and 7.4 for reliable descriptions
($r = 0.5$, $0.05$) and 2.1 and 4.4 at $p = 0.9$.

## Outputs of `evaluate.py`

- `<name>_regret.csv`: regret after $n$ examples, network and Bayes, with
  and without a description, in units of the noise variance.
- `<name>_ess.csv`: ESS of the description.
- `<name>_trust.csv`: the trust $q$ at which the Bayes learner's prediction
  is closest to the network's.

## Validation

`tests/test_meta.py` checks the prompt layout, and that the prediction
ignores hidden rows and the order of the examples. A
200-step smoke run of `train.py` and `evaluate.py` completed on 2026-09-29.
