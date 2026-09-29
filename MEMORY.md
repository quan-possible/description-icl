# MEMORY

Last condensed: 2026-09-29

<!-- Bounded historical index. Read STATUS.md for current state. -->
<!-- Oldest to newest; older spans broader, recent spans finer. -->
<!-- For detail, inspect dated files within the relevant section range. -->
<!-- Target: 100 lines / 8 KiB. Maximum: 200 lines / 16 KiB. -->

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, Clear Sans, Noto Sans, Helvetica Neue, Arial, Noto Sans CJK JP, sans-serif","fontSize":"16px","primaryTextColor":"#171717","lineColor":"#a8a8a8","mainBkg":"#ffffff","clusterBkg":"#ffffff","clusterBorder":"#ffffff","edgeLabelBackground":"#ffffff"},"flowchart":{"htmlLabels":true,"curve":"linear"}}}%%
flowchart TD
    classDef green fill:#edf3e8,stroke:#8a9f7a,color:#171717,stroke-width:1px
    classDef edgeGreen stroke:#8a9f7a,color:#8a9f7a,stroke-width:2px

    sep["2026-09 · Proposal written"] e1@--> setup["2026-09-29 · Repo and<br/>project records set up"]

    class sep,setup green
    class e1 edgeGreen
```

## 2026-09-29 | Project set up

- The project started from a finished proposal,
  [Task Descriptions as Bayesian Priors](docs/proposal/bayesian_icl/bayesian_icl_proposal.tex)
  (September 2026). Its preliminary $p = 1$ results were computed outside this
  repository, so they must be reproduced before they count as project
  evidence.
- The folder became a Git repository pushed to private GitHub
  `quan-possible/descriptor-icl`, with core project records and layout
  conventions in `AGENTS.md`.

## Rebuild rule

- Rebuild from dated records and the files that own each claim. When over
  budget, condense the oldest adjacent periods first.
