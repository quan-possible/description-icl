# Agent instructions for description-icl

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, Clear Sans, Noto Sans, Helvetica Neue, Arial, Noto Sans CJK JP, sans-serif","fontSize":"16px","primaryTextColor":"#171717","lineColor":"#a8a8a8","mainBkg":"#ffffff","clusterBkg":"#ffffff","clusterBorder":"#ffffff","edgeLabelBackground":"#ffffff"},"flowchart":{"htmlLabels":true,"curve":"linear"}}}%%
flowchart TD
    classDef gray fill:#f5f5f5,stroke:#b3b3b3,color:#171717,stroke-width:1px
    classDef green fill:#edf3e8,stroke:#8a9f7a,color:#171717,stroke-width:1px
    classDef pink fill:#f5e7ec,stroke:#bd7c8f,color:#171717,stroke-width:1px
    classDef blue fill:#e8f0f9,stroke:#7f9fc4,color:#171717,stroke-width:1px
    classDef region fill:#fafafa,stroke:#ececec,color:#5a5a5a,stroke-width:1px
    classDef edgeGreen stroke:#8a9f7a,color:#8a9f7a,stroke-width:2px
    classDef edgeBlue stroke:#7f9fc4,color:#7f9fc4,stroke-width:2px
    classDef edgePink stroke:#bd7c8f,color:#bd7c8f,stroke-width:2px

    proposal["Proposal<br/>docs/proposal/"]

    subgraph exact["Bayes-optimal learner (exact)"]
        rq1["RQ1 · Dimensionality<br/>single-query vs long-horizon ESS"]
        rq2["RQ2 · Reliability<br/>tipping points, mis-set trust"]
    end

    subgraph nets["Trained networks"]
        rq3["RQ3 · Meta-trained models<br/>match the Bayes-optimal ESS?"]
    end

    paper["Paper<br/>docs/paper/"]

    proposal e1@--> rq1
    proposal e2@--> rq2
    rq1 e3@--> rq3
    rq2 e4@--> rq3
    rq3 e5@--> paper

    class proposal gray
    class rq1,rq2 green
    class rq3 blue
    class paper pink
    class exact,nets region
    class e1,e2 edgeGreen
    class e3,e4 edgeBlue
    class e5 edgePink
```

## Goal

Measure how many in-context examples a task description is worth. Treat ICL
as Bayesian inference over a latent task: examples are data and the
description sets the prior. A description's worth is its **effective sample
size (ESS)**, which depends on its **precision** (how narrowly it constrains
the task) and its **reliability** (how often it is correct).

The testbed is in-context linear regression with a hierarchical Gaussian task
prior (precision) and a robust mixture prior (reliability), where the
Bayes-optimal learner is exact. The questions are fixed by the
[proposal](docs/proposal/bayesian_icl/bayesian_icl_proposal.tex). Claims apply
to Bayes-optimal predictors and small meta-trained sequence models, not to
large language models.

## Start here

- Read `STATUS.md` for the current position and next action.
- Read `MEMORY.md` and the newest `memory/YYYY-MM-DD.md` only when resuming
  work or reconstructing history.
- Create `docs/jobs/active/<slug>/JOB.md` only when work in this conversation
  will need later resumption. Do not reopen another conversation's job unless
  Bruce asks.

## Owners

| Path | Owns |
| --- | --- |
| `docs/proposal/bayesian_icl/` | The research proposal (LaTeX source and compiled PDF). |
| `README.md` | Human-facing overview, layout, and build steps. |
| `STATUS.md` | Current state, priorities, and next actions. |
| `MEMORY.md`, `memory/` | Project history. |

## Where new work goes

Create each folder when its first real file arrives; do not add empty
scaffolding.

| Role | Home |
| --- | --- |
| Shared code: priors, Bayes-optimal predictors, regret, ESS, models | `src/description_icl/`, tests in `tests/` |
| One analysis: its script or notebook, config, figures, tables, and a `README.md` stating what it shows | `results/<rq>-<slug>/`, e.g. `results/rq1-dimension-sweep/` |
| Hand-written prose: proposal, notes, paper | `docs/` (`docs/proposal/`, `docs/paper/`) |
| Resumable multi-session work | `docs/jobs/active/<slug>/JOB.md` |
| Disposable work | `tmp/` (ignored) |

- Analysis code that a second analysis needs moves into `src/`.
- Every number or figure used in writing must regenerate from its committed
  `results/` script with a fixed seed. Keep checkpoints and large arrays out
  of Git.
- Keep the proposal as the record of the original plan; write the paper
  separately.

## Research rules

- Validate every numerical routine against a closed form or a known limit
  before using it for claims (for example, the long-horizon ESS limit, $d = 1$,
  and balanced designs $X^\top X \propto I$).
- Report the setting (dimension, noise, hierarchy scales, horizon, $p$) beside
  every ESS number.
- Keep cited results, derivations, hypotheses, planned experiments, and actual
  findings distinct in writing and records.
- If a result contradicts the proposal's hypotheses, record it in `STATUS.md`
  and the dated memory; do not quietly reframe the question.

## Close-out

- Commit completed work locally. Pushing to the private GitHub remote
  `quan-possible/description-icl` needs Bruce's approval unless he asked for it.
- No Drive replication or deployment is configured.
