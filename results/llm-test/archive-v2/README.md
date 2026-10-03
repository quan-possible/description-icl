# The language-model test (paper Section 7)

**What it asks.** What is a task description worth, in demonstrations, to a
pretrained language model that reads it in words, with no training? The theory
gives the worth for an ideal learner; this measures it for a real one.

## Setting

The paper's model in its simplest case: one hidden mean `w ~ N(500, 100^2)` and
numbers `w + N(0, 30^2)`, rounded to integers and kept inside 100 to 999. A
prompt is 200 numbers, one per line, with one optional line on top:

```
Mean: 508 ± 10
516
497
523
...
```

| Condition | Top line |
| --- | --- |
| `none` | no line |
| `valid30`, `valid10`, `valid3` | `Mean: m ± tol`, with `w` within `tol` of `m` (ideal worth 0.9, 8.9, 99.9 numbers) |
| `exact` | `Mean: m` with `m = w` |
| `invalid30`, `invalid10`, `invalid3` | the same line with the value of an independent task |
| `valid*_sentence`, `invalid*_sentence` | `The mean is about m, give or take tol.` |
| `valid*_interval`, `invalid*_interval` | `Mean in [m - 2 tol, m + 2 tol]` |

Every condition uses the same 2,000 tasks (seed 0). There is no preamble, no
instruction and no chat template.

## Measurement

Every three-digit integer is one token for these models. One forward pass gives
the next-token distribution at all 200 positions; restricted to those 900
tokens and renormalized, it is the model's predictive distribution for the
next number, given that a number comes next. From each distribution we keep
the regret (expected log loss under the true distribution of the number, minus
the oracle's), its mean and standard deviation, and the probability the model
put on the number tokens before renormalizing (`mass`). That probability is
low before the first numbers and, with a line present, dips again around the
fifth number, so every ESS is reported two ways: over the number tokens (the
saved regret) and over all tokens (the regret minus `log(mass)`).

- **Implied strength** (the main readout): at each n, regress the mean of the
  predictive distribution on the stated value and on the mean of the numbers so
  far, with a constant: `mean_n = a_n m + b_n ybar_n + const`. The strength is
  `n a_n / b_n`, averaged over 5 to 10 numbers in hand: the number of numbers
  the model weighs the line as. The ideal learner has `c` for a valid line.
- **ESS against no line**: Definition 1 of the paper with the model's own two mean
  regret curves (with and without the line; the latter made monotone), averaged
  over the same window, with a bootstrap interval over tasks.
- **Content ESS**: the same with the invalid line of the same wording and
  tolerance as the reference curve, which holds the form of the prompt fixed.
  It measures what the line's content adds (the gain from the right value plus
  the harm from the wrong one); it is not a bound on the ESS against no line.
  Also computed in later windows.
- `pstated`: the probability the model puts on exactly the stated number.

## Run

```bash
python3 results/llm-test/llm_test.py                 # self-check on the ideal learner
# on a GPU (Colab A100; float16):
python3 score_model.py allenai/OLMo-2-1124-7B base7.npz --tasks 2000 --batch 16
python3 score_model.py allenai/OLMo-2-1124-7B-Instruct instruct7.npz --tasks 2000 --batch 16
# locally, from the downloaded arrays (7B, 13B and 32B, base and instruction-tuned):
cd results/llm-test && python3 analyze.py base7.npz instruct7.npz base13.npz instruct13.npz base32.npz instruct32.npz
```

`run_all.sh` and `run_size.sh` are the Colab job scripts. `analyze.py` writes
`summary.csv`, `curves.csv`, `bayes.csv` (the Bayes-optimal yardsticks), `llm_table.tex`,
`llm_table_full.tex` and `llm_test.pdf` (about 10 minutes for six models). `probe_other.py` and the two `probe_*.json` files
record what the 7B models expect instead of a number (first-run task set). `archive-v1/` holds the first
run, which had no invalid lines for the sentence and interval wordings. The power calculation that set
2,000 tasks is `docs/paper/figs/llm_test_power.py`.

## Files

- `llm_test.py`: tasks, prompts, the ideal learner, the estimators.
- `score_model.py`: scores a model; needs `torch` and `transformers`.
- `analyze.py`: summary, curves and figure.
