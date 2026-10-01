# RQ3: do meta-trained networks match the Bayes-optimal learner?

**What it shows.** Networks are Bayes-optimal within their output class.
They reproduce the Bayes-optimal ESS for reliable descriptions (14.7 vs 15.4
at `r` $= 0.01$; 5.24 vs 5.28 at `r` $= 0.1$). With an unreliable description
alone the Bayes-optimal predictive is a two-component mixture that a
mean-and-variance head cannot express; the networks match the best single
Gaussian instead (ESS 4.30 vs 4.31 at `r` $= 0.01$; 3.57 vs 3.62 at
`r` $= 0.1$; mixture 6.8 and 4.2; `single_gaussian.py`), and the three
merge as examples identify the description (`ess_n.py`). The implied trust
of 0.85 is a posterior-mean readout of that unimodality, not a belief.
Trust is inherited from training: models trained at $p = 1$ meet unreliable
descriptions as fully trusting learners (precise: ESS 0 vs calibrated 6.9),
and models trained at $p = 0.9$ keep hedging on reliable prompts (6.4 vs
15.6; `*_ptest1.0_*`). `trust_curve.py` gives the exact regret against the
learner's trust. An
earlier model at `r` $= 0.05$, $p = 0.9$ (files kept here) showed the same
under-trust and a continuation to 80k steps left it unchanged (3.9 vs 5.0,
trust 0.85), so it is not under-training. Models trained on reliable
descriptions do not discount an unreliable one: on $p = 0.9$ prompts the
precise model is hurt by a description (ESS 0 against a calibrated 6.9,
implied trust 0.999) and the coarse one keeps 2.3 of 4.3. See `summary.csv`,
`networks_table.tex`, and `regret.pdf`. Models trained 2026-10-01 on Colab
L4s (`one_model.sh`, 40k steps, prompts with 0 to 20 examples).

## Design

The design, a worked example, and the reasons are in
[docs/wiki/experiments.md](../../docs/wiki/experiments.md), section 6, which is authoritative.

| Quantity | Value |
| --- | --- |
| $d$, `a0` | 5, 10; the model sees $w \sim \mathcal{N}(0, I)$ and noise variance 0.1 |
| Prompt | Optional description, 0 to 20 examples, one question |
| `r` | 0.1, 0.01 (one model each; the earlier 0.5 and 0.05 models' files are kept) |
| $p$ | 1 in stage 1, 0.9 in stage 2 (one model each) |
| Model | Transformer, 12 layers, 8 heads, width 256 (Garg et al.), no positions |
| Output | A mean and a log variance for the answer to the question |
| Loss | Log loss (Genewein et al.) |
| Training | Adam, learning rate 1e-4, batch 1024; step count set by a pilot |
| Seeds | 0 (one model per configuration) |

## Run

```bash
uv run python results/rq3-meta-trained/targets.py    # the exact learner's targets
uv run python results/rq3-meta-trained/train.py --p 1.0 --r 0.05 --seed 0 --steps 40000 --amp
uv run python results/rq3-meta-trained/evaluate.py tmp/rq3/p1.0_r0.05_d5_s0.pt
uv run python results/rq3-meta-trained/evaluate.py tmp/rq3/p1.0_r0.05_d5_s0.pt --p-test 0.9
uv run python results/rq3-meta-trained/summarize.py  # summary.csv and the paper's table
uv run python results/rq3-meta-trained/figure.py     # regret.pdf
```

The four models were trained on Colab L4 GPUs on 2026-09-30 (`colab_job.sh`
runs the pilot and one model; `one_model.sh` runs one model; `grid.sh` runs
them all in sequence; `continue_model.sh` trains a model 40k steps further).
Checkpoints go to `tmp/rq3/` and stay out of Git. `evaluate.py --p-test`
evaluates at a reliability different from training. `pilot.csv` records the
pilot's gap to the exact learner at every 5k-step snapshot.

## Targets

`targets.csv` holds the exact learner's log-loss regret (nats) and ESS at
this setting; see the file for the values.

## Outputs of `evaluate.py`

- `<name>_regret.csv`: log-loss regret after $n$ examples, network and
  Bayes, with and without a description, in nats.
- `<name>_ess.csv`: ESS of the description.
- `<name>_trust.csv`: the trust $q$ at which the Bayes learner's prediction
  is closest to the network's.

## Validation

`tests/test_meta.py` checks the prompt layout, and that the prediction
ignores hidden rows and the order of the examples. A
200-step smoke run of `train.py` and `evaluate.py` completed on 2026-09-29.
