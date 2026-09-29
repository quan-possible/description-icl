# JOB: Run the project end-to-end toward NeurIPS 2027

Scientific state lives in [RESEARCH.md](RESEARCH.md). This file holds only
what is needed to resume the task.

## Conversation

- Created 2026-09-29 in Claude Code session
  `session_018J9xvUNKRpF8XCYezNRrba`.

## Task

- **Result:** the best paper-level result attainable for the project's goal
  (what a task description is worth in examples), developed to a NeurIPS 2027
  submission draft in `docs/paper/`.
- **Done when:** no further feasible research move is worth its cost, the
  standing reviewer's final checkpoint is adjudicated, and the draft
  regenerates every number from `results/`.
- **Limits:** claims apply to Bayes-optimal predictors and small meta-trained
  models. The proposal stays unchanged as the record of the original plan.
- **Permissions (Bruce, 2026-09-29, this conversation):** run the project
  end-to-end locally. Paid LLM or external-compute calls and pushing to
  GitHub still need his approval.

## Instructions

- [AGENTS.md](../../../../AGENTS.md) (project contract, research rules).
- Global skills: `autonomous-research`, `subagent`, `instructions`, `memory`,
  `verification`, `writing`.

## Current state

- Environment: `uv` project, Python 3.12; `uv run pytest` passes 10 checks.
- Built: `src/descriptor_icl/gaussian.py` (reliable descriptions) and
  `mixture.py` (unreliable descriptions), both exact up to Monte Carlo over
  inputs.
- Analyses: `results/rq1-single-query-gap/`, `results/rq2-reliability/`.
- Next action: see "Next moves" in [RESEARCH.md](RESEARCH.md).

## History

- 2026-09-29: Campaign opened. Chose to rebuild the exact learner rather than
  search for the proposal's lost preliminary code.
