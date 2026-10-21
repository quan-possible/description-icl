# RQ3: do meta-trained networks match the Bayes-optimal learner?

**What it shows.** Scored by the squared error of their mean prediction, as
in the paper, the networks come close to the Bayes-optimal ESS at their
training reliability (specific description, mean ± sd over three seeds:
14.3 ± 0.5 vs 15.9 at $p = 1$, 9.3 ± 0.2 vs 9.0 at $p = 0.99$, 4.66 ± 0.01 vs
4.70 at $p = 0.9$; loose, two seeds: within 0.1), and take their trust from
training. Trained at $p = 0.9$ they keep hedging on reliable prompts
(9.5 ± 0.4 vs 15.9; implied trust 0.87 to 0.89); trained at $p = 1$ or 0.99
they trust fully (0.95 to 1.02, the two not distinguishable over seeds). Trained at $p = 1$ they trust fully: on
$p = 0.9$ prompts that costs nothing with no examples (4.61 vs 4.70) but from
6 examples on the description makes the error worse than none (4.6 vs 2.3 at
$n = 8$), as the fully trusting Bayes learner's does. See `summary.csv`,
`ess_n.csv` and the paper's Table 1 (`summarize.py` writes it to
`docs/paper/figs/networks_table.tex`). Models trained 2026-10-01 on Colab L4s
(`one_model.sh`, 40k steps, prompts with 0 to 20 examples); rescored in
squared error on 2026-10-19 (Colab L4, about 40 minutes).

Before the rescoring, every output here was in log loss, which also judges
the predicted variance: with $p < 1$ the networks then matched the best single
Gaussian rather than the Bayes-optimal mixture (`single_gaussian.py`,
`trust_curve.py`, both still log loss). Those results are in Git history
before the squared-error rescoring commit.

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
| Seeds | 0, 1, 2 for $r = 0.01$; 0, 1 for $r = 0.1$ (seeds 1 and 2 added 2026-10-20, `run_seeds.sh`) |

## Run

```bash
uv run python results/rq3-meta-trained/targets.py    # the exact learner's targets
uv run python results/rq3-meta-trained/train.py --p 1.0 --r 0.05 --seed 0 --steps 40000 --amp
uv run python results/rq3-meta-trained/evaluate.py tmp/rq3/p1.0_r0.05_d5_s0.pt
uv run python results/rq3-meta-trained/evaluate.py tmp/rq3/p1.0_r0.05_d5_s0.pt --p-test 0.9
uv run python results/rq3-meta-trained/summarize.py  # summary.csv and the paper's Table 1
uv run python results/rq3-meta-trained/figure.py     # regret.pdf
```

The four models were trained on Colab L4 GPUs on 2026-09-30 (`colab_job.sh`
runs the pilot and one model; `one_model.sh` runs one model; `grid.sh` runs
them all in sequence; `continue_model.sh` trains a model 40k steps further).
Checkpoints go to `tmp/rq3/` and stay out of Git. `evaluate.py --p-test`
evaluates at a reliability different from training. `pilot.csv` records the
pilot's gap to the exact learner at every 5k-step snapshot.

## Targets

`targets.csv` holds the exact learner's squared error and ESS at this
setting; see the file for the values.

## Outputs of `evaluate.py`

- `<name>_regret.csv`: squared error of the prediction after $n$ examples
  against $w^\top x$, network (its mean) and Bayes, with and without a
  description.
- `<name>_ess.csv`: ESS of the description.
- `<name>_trust.csv`: the trust $q$ at which the Bayes learner's prediction
  is closest to the network's.

## Validation

`tests/test_meta.py` checks the prompt layout, and that the prediction
ignores hidden rows and the order of the examples. A
200-step smoke run of `train.py` and `evaluate.py` completed on 2026-09-29.

**Third reliability and second seed (2026-10-01).** Two models at $p = 0.99$
(`p0.99_r0.01_d5_s0`, `p0.99_r0.1_d5_s0`) and a second seed of the precise
$p = 0.9$ model (`p0.9_r0.01_d5_s1`), each evaluated at all three
reliabilities. The precise $p = 0.99$ network lands on the best single
Gaussian again (ESS $8.9 \pm 0.2$; single Gaussian $8.54$, mixture $12.88$),
the coarse one is within noise of both ($5.01$; $4.93$, $5.08$), and both read
an implied reliance of $0.95$. Tested at $p = 0.9$, the precise $p = 0.99$
model gets nothing from the description ($0.1 \pm 0.4$ against $6.9$
calibrated). Seed 1 reproduces seed 0 ($4.26 \pm 0.08$ against $4.30$ alone;
$6.6$ against $6.4$ on $p = 1$ prompts; reliance $0.9$ against $0.85$ on a grid
of step $0.05$). `networks_table.tex` now takes every reference from
`targets.csv` (200,000 prompts, with ESS standard errors) and
`single_gaussian.csv` (100,000 prompts, $p \in \{0.9, 0.99\}$), gives the
network ESS a standard error (both regrets' standard errors over the local
slope of its demonstrations-only curve), and has a reliance column.

**More seeds (2026-10-20).** For the referee revision, seeds 1 and 2 of the
three $r = 0.01$ settings (seed 1 already existed at $p = 0.9$) and seed 1 of
the three $r = 0.1$ settings, trained on two Colab L4s (`run_seeds.sh`, two
`one_model.sh` queues per VM, about 3 hours per network), each evaluated at all
three reliabilities. `summarize.py` now averages Table 1 over seeds (± the
standard deviation across them) and takes trust from `trust_q.csv`, the closed
form of `trust_q.py` (the regression coefficient of the network's $n = 0$
prediction on $m^\top x$), which replaces the grid. Seed 0's trust of 0.95 at
$p = 0.99$ and 0.94 for the loose $p = 1$ network were seed effects: the other
seeds read 0.99 to 1.01.
