# The language-model test (paper Section 7 and Appendix C)

**What it asks.** Told everything the Bayes-optimal predictor knows, in words, how many demonstrations
is a description of stated specificity and stated reliability worth to a pretrained language model,
against its Bayes-optimal ESS?

## Setting

The paper's model in its simplest case: a hidden mean `w ~ N(500, 100^2)` and numbers
`w + N(0, noise^2)`, rounded to integers. The prompt is the user turn of the model's chat template:

```
The numbers below are drawn from a normal distribution with standard deviation 60.
Its mean was drawn from a normal distribution with mean 510 and standard deviation 30.

548
431

Predict the next number. Reply with the number only.
```

| Condition | Sentence about the mean |
| --- | --- |
| no description | `Its mean is unknown.` |
| relevant, sd `tau` | `Its mean was drawn from a normal distribution with mean m and standard deviation tau.` with `w \| m ~ N(m, tau^2)` |
| irrelevant | the same sentence with the value of an independently drawn task |
| stated reliability | one more sentence: `There is a 10% chance that this statement is wrong, in which case the mean is unknown.` (or 50%) |

The Bayes-optimal predictor knows only what the prompt says (noise, description; otherwise a mean
uniform over 100..999), so a relevant description's Bayes-optimal ESS is `c = (noise / tau)^2`
exactly and its weight on the stated value with `n` numbers shown is `c / (c + n)`. Noise 60 with
`tau` 60, 30, 10 gives `c` = 1, 4, 36; noise 30 with `tau` 30, 15, 5 gives the same `c` as a check.
19 conditions, `n` = 0, 1, 2, 4, 8, 16, 32, 64, 2,000 tasks each.

## Measurement

- **Distribution readout** (gpt-oss-20b, Phi-4, OLMo 2 32B; also MiniCPM5 2B, OLMo 2 7B/13B, which mostly
  explain instead of answering): every three-digit integer is one token, so the first-token distribution
  over the 900 number tokens, renormalized, is the model's distribution over the next number; its mean
  is the prediction.
- **Written readout** (`--written`; Qwen3.5 9B, Gemma 4 12B, Gemma 4 31B, and the three above as a
  check): the first three-digit integer the model writes under greedy decoding with 8 new tokens.
- **ESS**: Definition 1 with squared error, read off the model's own no-description error curve
  (from n = 1, made non-increasing, interpolated in log n); headline value = median over n = 1, 2, 4
  with a 95% bootstrap interval over tasks. **Weight**: regression of the prediction on the stated value
  and the mean of the numbers shown. Models that answer with a number in fewer than 75% of replies are
  not reported beyond that; cells below 75% are italic in the appendix tables.
- **Wording ablation** (`--paraphrase K --ablation`): three other wordings of the description sentence
  on the no-description and three relevant conditions at n = 1, 2, 4 (written readout). `written_*_p0.npz`
  are the paper's wording masked to the same grid.
- **Reasoning** (`reason`): gpt-oss-20b at low reasoning effort, 100 tasks, 7 conditions, n = 1 and 4;
  `reason_table.py` writes `reason_table.tex`.
- **Frontier model**: `agent-check/` (GPT-5.6-Luna through the Codex CLI, reasoning low and off; see its README).

## Run

```bash
python3 llm_test.py          # example prompts and a self-check of the ideal learner
# on a Colab GPU (run_all.sh: the six distribution-readout models; run_written.sh: the written readout and the
# wording ablation on six models; run_reason.sh: the reasoning run):
python3 llm_test.py score openai/gpt-oss-20b full_gptoss20.npz --tasks 2000
python3 llm_test.py score Qwen/Qwen3.5-9B written_qwen9.npz --tasks 2000 --written
python3 llm_test.py score Qwen/Qwen3.5-9B written_qwen9_p1.npz --tasks 2000 --written --paraphrase 1 --ablation
# locally:
python3 analyze.py full_gptoss20.npz full_phi4.npz full_olmo32.npz full_minicpm2.npz full_olmo7.npz full_olmo13.npz \
    written_gptoss20.npz written_phi4.npz written_qwen9.npz written_gemma12.npz written_gemma31.npz written_olmo32.npz written_*_p[0-3].npz
python3 analyze.py replot     # tables and figure from the cached results
python3 reason_table.py
```

`analyze.py` writes `summary.csv`, `medians.csv`, `llm_table.tex` (Table 2, all compliant models),
`llm_table_worth.tex` and `llm_table_full.tex` (every n, the distribution-readout models),
`llm_table_paraphrase.tex` (the wording ablation) and `llm_test.pdf` (Figure 4, the three distribution-readout
models). The paper inputs these from `docs/paper` via `../../results/llm-test/`.

## Files

- `llm_test.py`: tasks, prompts, paraphrases, the Bayes-optimal predictor, scoring (distribution / written / reasoning), read-out.
- `analyze.py`: summary, tables, figure. `reason_table.py`: the reasoning table.
- `full_*.npz`: distribution readout (64 numbers shown); `written_*.npz`: written readout; `written_*_p[0-3].npz`: wording ablation; `reason_gptoss20.npz`.
- `run_all.sh`, `run_written.sh`, `run_reason.sh`, `run_olmo32.sh`: the Colab jobs.
- `agent-check/`: the frontier-model check.
- `archive-n8/`: the earlier 8-number run; `archive-pilot1/`, `archive-pilot2/`: prompt pilots; `archive-v2/`: the earlier list-continuation version of the test.
- `tmp/`: job logs and the pre-merge analysis script.
