# descriptor-icl

**Task descriptions as Bayesian priors: how many in-context examples is an
instruction worth?**

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, Clear Sans, Noto Sans, Helvetica Neue, Arial, Noto Sans CJK JP, sans-serif","fontSize":"16px","primaryTextColor":"#171717","lineColor":"#a8a8a8","mainBkg":"#ffffff","clusterBkg":"#ffffff","clusterBorder":"#ffffff","edgeLabelBackground":"#ffffff"},"flowchart":{"htmlLabels":true,"curve":"linear"}}}%%
flowchart TD
    classDef gray fill:#f5f5f5,stroke:#b3b3b3,color:#171717,stroke-width:1px
    classDef green fill:#edf3e8,stroke:#8a9f7a,color:#171717,stroke-width:1px
    classDef pink fill:#f5e7ec,stroke:#bd7c8f,color:#171717,stroke-width:1px
    classDef blue fill:#e8f0f9,stroke:#7f9fc4,color:#171717,stroke-width:1px
    classDef edgeGreen stroke:#8a9f7a,color:#8a9f7a,stroke-width:2px
    classDef edgeBlue stroke:#7f9fc4,color:#7f9fc4,stroke-width:2px
    classDef edgePink stroke:#bd7c8f,color:#bd7c8f,stroke-width:2px

    desc["Task description<br/>precision · reliability"] e1@--> prior["Prior over the task"]
    ex["In-context examples"] e2@--> post["Posterior"]
    prior e3@--> post
    post e4@--> regret["Regret over the<br/>next N predictions"]
    regret e5@--> ess["ESS: examples that match<br/>the description's regret"]

    class desc,prior green
    class ex,post blue
    class regret gray
    class ess pink
    class e1,e3 edgeGreen
    class e2 edgeBlue
    class e4,e5 edgePink
```

Theories of in-context learning mostly study prompts made only of examples.
Real prompts also carry an instruction. This project models the instruction
as the prior in Bayesian inference over a latent task and measures its worth
as an **effective sample size**: the smallest number of examples whose regret
is no larger than the description's alone.

The setting is in-context linear regression, where the Bayes-optimal learner
is exact. A hierarchy of Gaussian task priors varies how *precise* a
description is; a robust mixture prior varies how *reliable* it is. The
research questions are:

1. **RQ1:** How does task dimensionality change the gap between single-query
   and long-horizon ESS?
2. **RQ2:** What does an unreliable or mis-trusted description cost?
3. **RQ3:** Do meta-trained Transformers and LSTMs reproduce the Bayes-optimal
   ESS?

The full plan is in the
[proposal](docs/proposal/bayesian_icl/bayesian_icl_proposal.pdf).

## Layout

| Path | Contents |
| --- | --- |
| `docs/proposal/bayesian_icl/` | Research proposal (LaTeX source and PDF). |
| `AGENTS.md` | Project contract for agents, including where code, experiments, results, and the paper go. |
| `STATUS.md` | Current state and next actions. |
| `MEMORY.md`, `memory/` | Project history. |

## Build the proposal

```bash
cd docs/proposal/bayesian_icl && latexmk -pdf bayesian_icl_proposal.tex
```

This needs a TeX distribution with `titlesec` and `enumitem`. A BasicTeX
install lacks both; add them with:

```bash
sudo tlmgr install titlesec enumitem
```
