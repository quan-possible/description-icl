# The filing-policy task

A semantic counterpart of the paper's language-model test, proposed in an external review and adapted here. A company
files messages into folder A or B by a hidden policy. A message has six binary properties written in ordinary language:
outside sender, payment request, travel, computer access, a deadline within a day, and contact details. A policy uses
two properties and requires each to be present or absent, which gives 60 policies. The prompt states this family and
that the policy is uniform over it, so the Bayes-optimal probability of A is a count over the policies consistent with
the prompt. Code: `filing.py`.

**Descriptions:**
- **None.**
- **Broad:** names one property (20 policies left).
- **Specific:** gives one property and its required value, plus the other property (2 left).
- **Unreliable:** a specific note that the prompt says has a 50% chance of being wrong. It is in fact wrong half the
  time, when it describes an independent policy, and the prompt says that a wrong note leaves the policy uniform over
  the 60. Unlike the paper's original stated-reliability condition, the fallback is fully stated, so the reference is
  exact.

**Measurement.** 200 tasks (policy draws). Each prompt has 4 query messages; the model gives P(A) for each, scored by
squared error (Brier) against the true folder. The ESS of each note is read against the same learner's own curve without
a note at n = 0, 2, 4, 6, 8, 12, 16, 20, 24, 32 (Definition 1, with the monotone reading as a check, as in the paper's
Table 4). On 3,000 draws, interpolating this grid moves the oracle's ESS by at most 0.6 from its value on every n; the
pilot's coarser grid (0, 2, 4, 8, 16, 32) overstated the specific note's ESS by up to 3.6. For the unreliable note, the
report also gives the share of queries closer to the updating oracle than to the fixed-chance rule (keep the stated 50%).

**Oracle checks** (3,000 simulated tasks): the oracle is calibrated in every cell, and its posterior that the
unreliable note is right matches how often it is (0.82 against 0.82 where it exceeds 0.7).

**Runs** (`run.sh`, in tmux, log `run.log`):
- GPT-6 Sol without reasoning and at low effort (Codex CLI, subscription; Pro account from 19:50), 20 calls at a time.
- Claude Haiku 5.5 at medium effort (Claude Code CLI, subscription), 8 at a time.
- Both harnesses (`../llm-test/sol-check/ask.sh`, `../llm-test/haiku/ask.sh`) make bare calls with no tools.
- Replies go to `replies-*`. The Haiku folder keeps the pilot's 360 replies, which are the same prompts (tasks 0–23).
- `python3 report.py replies-sol-none replies-sol-low replies-haiku-medium` scores them: losses and ESS with 95%
  bootstrap intervals over tasks, rule-following, missing replies, and the share of replies that reasoned.

**Property check** (with the pilot): GPT-6 Sol without reasoning reads each message's six properties correctly in
97–100% of 96 messages, so the wordings are unambiguous.

## Results (200 tasks, 3,800 prompts per setting, run 4 November 2026; `report.txt`)

Every reply is in. 14 Sol low-effort replies and 1 Haiku reply lack a probability, and no Sol reply called a tool. The
monotone reading gives the same ESS as Definition 1 in every cell.

**ESS of each note**, against the same learner's own curve without a note (95% bootstrap intervals over tasks in
`report.txt`). "≤ 0" means the note did not help; "> 28" means the no-note curve never reaches the note's loss by n = 32.

| Note, n | Oracle | Haiku, medium | Sol, low effort | Sol, no reasoning |
| --- | --- | --- | --- | --- |
| Broad, 0 | 0.0 | 0.0 | ≤ 0 | ≤ 0 |
| Broad, 2 | 1.4 | 1.3 | 1.3 | 0.7 |
| Broad, 4 | 1.8 | 1.6 | 1.7 | 8.6 |
| Specific, 0 | 3.7 | 3.6 | 3.3 | 3.7 |
| Specific, 2 | 11.4 | 9.9 | 9.5 | 13.5 |
| Specific, 4 | 14.8 | 14.5 | 26.1 | > 28 |
| Unreliable, 0 | 1.3 | 1.3 | 1.6 | 1.7 |
| Unreliable, 2 | 3.0 | 2.6 | 3.2 | 1.7 |
| Unreliable, 4 | 5.0 | 4.9 | 5.1 | 2.7 |

**Brier loss** with the specific note at n = 0, 2, 4, and with no note at n = 32:

| | Oracle | Haiku, medium | Sol, low effort | Sol, no reasoning |
| --- | --- | --- | --- | --- |
| Specific note | 0.122, 0.027, 0.007 | 0.125, 0.036, 0.008 | 0.134, 0.049, 0.008 | 0.173, 0.130, 0.064 |
| No note, n = 32 | 0.000 | 0.000 | 0.005 | 0.078 |

**Unreliable note.** On the queries where the updating oracle and the fixed-chance rule differ by at least 0.05, the share
of replies closer to the oracle: Haiku 90% at n = 2 and 99% at n = 4; Sol at low effort 77% and 87%; Sol without
reasoning 45% and 45%.

**Reading.**
- With reasoning, both models use each note almost as the oracle does: the ESS is within 2 of the oracle's
  except one cell, and with the fallback stated they update the note's reliability from the demonstrations.
- In that cell, Sol's specific note at n = 4, the ESS is 26.1 against the oracle's 14.8. Sol's own no-note curve
  flattens past n = 16 (0.022 at n = 20 against the oracle's 0.004), so it takes many more of its own demonstrations to
  match the note.
- Without reasoning, Sol learns slowly from demonstrations: its loss without a note is still 0.078 at n = 32. So in its
  own unit a note is worth as much as or more than to the oracle (13.5 for the specific note at n = 2, interval 3.7 to
  21.9), although in loss it uses the note far worse (0.130 against 0.027). It keeps the stated reliability rather than
  updating it.
- The pilot's 2.4 for Sol without reasoning came from a no-note curve partly computed in Python.

## The pilot (24 tasks; `archive/pilot/`)

The pilot found the pattern the full run tests: with reasoning, Sol and Haiku used the specific note almost as the
oracle does; without reasoning, Sol barely used it; and with the fallback stated, the reasoning models updated the
unreliable note's reliability from the demonstrations. Its numbers are superseded, because a check before the full run
found three faults in it:
- **Fractions.** The reply parser read "1/3" and "13/36" as 1 and dropped "5/36". This touched 27 Sol replies at low effort
  and 15 without reasoning. `filing.read` now reads decimals, fractions and percentages.
- **Tools.** The Codex harness gave Sol a shell, and Sol computed its answer in Python in 160 of 360 replies without
  reasoning and 84 of 360 at low effort. The harness now makes a bare call, and `report.py` counts any reply that
  called a tool as missing. The same fault touched the paper's Sol runs (`../llm-test/tool_use.py`).
- **Grid.** The coarse no-note grid overstated the ESS, as above.
