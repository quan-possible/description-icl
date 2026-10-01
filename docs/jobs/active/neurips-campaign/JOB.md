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

- Environment: `uv` project, Python 3.12; `uv run pytest` passes.
- Built: `src/descriptor_icl/gaussian.py` (reliable descriptions) and
  `mixture.py` (unreliable descriptions), both exact up to Monte Carlo over
  inputs.
- Analyses: `results/rq1-single-query-gap/`, `results/rq2-reliability/`;
  network pipeline in `results/rq3-meta-trained/`, untrained.
- Paper: `docs/paper/paper.tex`, simplified on 2026-10-01: two propositions
  (additive precision rule; reliability sandwich with $n$ in hand),
  definitions in the sources' form, Figure 2 carrying $n$, $r$, $p$, tables
  in Appendix A, a fully trusting learner in Section 5. Section 6 describes
  the four existing models; retraining on the new grid waits for Bruce.
- Next action: see "Next moves" in [RESEARCH.md](RESEARCH.md).

## History

- 2026-09-29: Campaign opened. Chose to rebuild the exact learner rather than
  search for the proposal's lost preliminary code.
- 2026-09-30: The paper became the plan; Propositions 1 and 2 written out;
  the design record moved to `docs/wiki/experiments.md`.
- 2026-10-01: Powers-of-ten grids; adversarial, number, and grounding
  reviews; simplification pass (two propositions, sources' definitions,
  prose cut). Network retraining relaunched and stopped on Bruce's call.
