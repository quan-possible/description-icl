# A frontier model inside an agent harness (paper Appendix, Table "agent")

The paper's prompts for relevant descriptions with Bayes-optimal ESS c = 1, 4, 36 (noise 60; tau = 60, 30, 10)
at n = 1 and 4 demonstrations, the first 100 tasks of the main run (`prompts.json`, generated from `../llm_test.py`),
sent one at a time to GPT-5.6-Luna through the Codex CLI on a ChatGPT subscription (no API spend):
`codex exec -m gpt-5.6-luna -s read-only --ephemeral`, user instructions and notifications off. The harness still wraps the
60-token prompt in about 26,000 tokens of system prompt and tool definitions; the model used no tool in any of these replies.
Two passes over the same prompts: reasoning effort `low` (about 115 reasoning tokens per answer; `replies/`, `logs/`)
and `none` (`replies-none/`, `logs-none/`). All 1,200 replies are bare numbers (181 with a decimal part, kept).

- `ask.sh K` sends prompt K of `$PROMPTS` (default `prompts.json`; env `EFFORT`); `run.sh [effort]` sends them all, six at a time.
- `prompts_nodesc.json`: the no-description prompts on the same 100 tasks at n = 1, 2, 4, 8, 16, 32, 64 (700), both efforts
  (`replies_nodesc/`, `replies_nodesc-none/`), the no-description curve from which the model's own squared-error ESS is read.
  With reasoning `low` the model averaged the numbers with a shell command (Python in 127, Node.js or awk in 47) in 174 of the 700 no-description prompts
  (4 at n = 16, 74 at n = 32, 96 at n = 64; `command_execution` items in `logs_nodesc/`); with `none` it used no tool.
- `analyze_check.py [replies-dir]` prints weight on the stated value, implied worth n*a/b, and RMSE against the
  Bayes-optimal predictor; with both passes present it writes `agent_table.tex`, and with the no-description replies present `agent_ess_table.tex`
  (Definition 1 with squared error at n = 1 and 4, as in `../analyze.py`, with bootstrap intervals over tasks).

Result (paper Tables "agent" and "agent-ess"): with reasoning, the weight on the stated value is the Bayes-optimal c/(c+n)
to within 0.03 in all six cells and the ESS read from the model's own no-description curve matches the Bayes-optimal
predictor's read the same way (32 vs 33 for c = 36; within 0.2 for c = 1, 4). With reasoning off the weight is flat
(0.72-0.88 at n = 1, 0.40-0.47 at n = 4) and the c = 36 description is worth 5.4 demonstrations at n = 1 and 1.3 at n = 4.
Each pass of 600 or 700 prompts took 10-13 minutes at six in parallel.
