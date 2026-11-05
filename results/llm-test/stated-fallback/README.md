# Stated reliability with the fallback stated (4 November 2026)

The paper's stated-reliability condition, rerun with one change. The original warning was "There is a {q}% chance that
this statement is wrong, in which case the mean is unknown." That leaves open what the mean could be if the
description is wrong, and the Bayes-optimal reference assumed a uniform over 100..999. The new warning states that
fallback: "…in which case the mean was instead drawn uniformly at random from the whole numbers 100 to 999." Everything
else is the paper's design: the same 200 tasks, the c = 36 description, stated 10% and 50%, and n = 1, 2, 4.
`make_prompts.py` writes the 1,200 prompts and `run.sh` runs them. The models are GPT-6 Sol at low effort (Codex CLI)
and Claude Haiku 5.5 at medium effort (Claude Code CLI), both on the subscription. `compare.py` compares the runs with
the paper's originals.

**Result** (stated 50%): weight on the stated value, from the paper's regression, and the share of tasks closer to the
Bayes-optimal prediction than to the fixed-chance rule p·36/(36 + n), among the 123–155 tasks per cell where the two
differ by at least 5.

| Model | n | Bayes-optimal | Fixed rule | Original: weight / closer | Fallback stated: weight / closer |
| --- | --- | --- | --- | --- | --- |
| GPT-6 Sol, low | 1 | 0.58 | 0.49 | 0.50 / 2% | 0.57 / 96% |
| GPT-6 Sol, low | 2 | 0.61 | 0.47 | 0.48 / 1% | 0.58 / 97% |
| GPT-6 Sol, low | 4 | 0.64 | 0.45 | 0.45 / 0% | 0.58 / 89% |
| Haiku 5.5, medium | 1 | 0.58 | 0.49 | 0.55 / 41% | 0.62 / 95% |
| Haiku 5.5, medium | 2 | 0.61 | 0.47 | 0.56 / 30% | 0.60 / 91% |
| Haiku 5.5, medium | 4 | 0.64 | 0.45 | 0.46 / 21% | 0.63 / 87% |

With the stated 10% chance, at most 7 tasks per cell separate the two rules, so that condition is uninformative.

**What it means.** With reasoning, models do check a description against the demonstrations when the prompt says what
a wrong description implies. When the prompt leaves it open ("the mean is unknown"), they keep the stated chance fixed,
for which the correct update is undefined. The paper's claim that models "take a stated reliability at face value"
should be revised to this. The filing-policy pilot (`../../filing-policy/`) shows the same on semantic rules.

**ESS** (`ess.py`, computed as for Table 8, against each model's original curve without the description):

| Model | Stated chance | Original | Fallback stated | Bayes-optimal |
| --- | --- | --- | --- | --- |
| GPT-6 Sol, low | 50% | 4.4 | 4.7 | 5.7 |
| GPT-6 Sol, low | 10% | 30.3 | 17.8 | 27.5 |
| Haiku 5.5, medium | 50% | 3.8 | 4.2 | 5.7 |
| Haiku 5.5, medium | 10% | 32.6 | 17.9 | 27.5 |

- **At 50%**, the switch in rule barely moves the ESS, because the two rules give similar errors at n = 1–4.
- **At 10%**, with the fallback stated, the models discount the description more than the Bayes-optimal predictor:
  GPT-6 Sol's weight at n = 4 is 0.75, against 0.84.

Written into the adopted paper (docs/paper/paper.tex): Section 6.3, Appendix C "Stating the fallback", Table 10.
