# RQ3: do meta-trained networks match the Bayes-optimal learner?

**What it shows.** Nothing yet. The pipeline runs end to end; no model has
been trained to completion.

The design and the reasons for it are in
[docs/DESIGN.md](../../docs/DESIGN.md), which is authoritative.

## Design

The design, a worked example, and the reasons are in
[docs/DESIGN.md](../../docs/DESIGN.md), section 6, which is authoritative.

| Quantity | Value |
| --- | --- |
| $d$, `a0`, $\sigma_y$ | 5, 10, 1 |
| Prompt | Optional description, 0 to 15 examples, one question |
| `r` | 0.5, 0.05 (one model each) |
| $p$ | 1, 0.9, 0.7 (one model each) |
| Model | Transformer encoder, 6 layers, width 128, 4 heads, no positions |
| Output | Mixture of two Gaussians for the answer to the question |
| Loss | Log loss on that answer |
| Training | AdamW, learning rate 3e-4, batch 1024; step count set by a pilot |
| Seeds | 0, 1, 2 |

## Run

```bash
uv run python results/rq3-meta-trained/train.py --p 0.7 --r 0.05 --seed 0
uv run python results/rq3-meta-trained/evaluate.py tmp/rq3/p0.7_r0.05_d5_s0.pt
```

Checkpoints go to `tmp/rq3/` and stay out of Git. `evaluate.py --p-test`
evaluates at a reliability different from training. Measured speed with the
earlier layout at batch 256: 54 steps/s on a Colab L4, 24 on a T4.

## Outputs of `evaluate.py`

- `<name>_regret.csv`: single-query regret after $n$ examples, network and
  Bayes, with and without a description.
- `<name>_ess.csv`: ESS of the description at horizons 1 and 8.
- `<name>_trust.csv`: the trust $q$ at which the Bayes learner's prediction
  is closest in KL to the network's.

## Validation

`tests/test_meta.py` checks the prompt layout, and that the prediction
ignores hidden rows and the order of the examples. A
200-step smoke run of `train.py` and `evaluate.py` completed on 2026-09-29.
