# A frontier model inside an agent harness (paper Appendix, Table "agent")

The paper's prompts for relevant descriptions with Bayes-optimal ESS c = 1, 4, 36 (noise 60; tau = 60, 30, 10)
at n = 1 and 4 demonstrations, the first 100 tasks of the main run (`prompts.json`, generated from `../llm_test.py`),
sent one at a time to GPT-5.6-Luna through the Codex CLI on a ChatGPT subscription (no API spend):
`codex exec -m gpt-5.6-luna -s read-only --ephemeral`, user instructions and notifications off. The harness still wraps the
60-token prompt in about 27,000 tokens of system prompt and tool definitions; the model used no tool in any reply.
Two passes over the same prompts: reasoning effort `low` (about 135 reasoning tokens per answer; `replies/`, `logs/`)
and `none` (`replies-none/`, `logs-none/`). All 1,200 replies are a bare three-digit number.

- `ask.sh K` sends prompt K (env `EFFORT`); `run.sh [effort]` sends all 600, six at a time.
- `analyze_check.py [replies-dir]` prints weight on the stated value, implied worth n*a/b, and RMSE against the
  Bayes-optimal predictor; with both passes present it writes `agent_table.tex`.
