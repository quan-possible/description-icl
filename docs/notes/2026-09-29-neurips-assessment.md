# NeurIPS 2027 assessment and design decisions (2026-09-29)

Record of one day's investigation: is the project novel, do its findings
hold, how have comparable papers fared, and what design follows. Numbers
come from `results/rq1-single-query-gap/` and `results/rq2-reliability/`.

## Verdict

The project has a real chance at NeurIPS 2027 if the paper is built on the
reliability results. My judgment is 35–45% if executed well and 15–20% as
the proposal frames it, against a base rate near 25%. These are judgment
figures, not estimates from data.

## How the evidence was gathered

| Inquiry | Method | Limits |
| --- | --- | --- |
| ICL theory novelty | Agent; web search and paper reading | Most papers read through page summaries |
| Statistics novelty | Agent; same | Several full texts blocked; minimum-description-length literature not searched in depth |
| LLM data | Agent; same | Table values not checked against PDFs |
| Venue outcomes | Agent; public copies of OpenReview data | OpenReview itself blocked; no ICML reviews |
| Reliability findings | `results/rq2-reliability/run.py`, run locally | Exact learner only |
| Huang & Ge setup, Reznik abstract, Majumdar abstract, Lin & Lee ICLR 2024 meta-review | Read directly | |

## Novelty

| Piece | Status | Closest source |
| --- | --- | --- |
| ESS defined by matching prediction regret against examples | Not found | Reimherr et al. 2021, Sec. 5.4, and Neuenschwander et al. 2020 name prediction-based and multi-parameter ESS as open |
| Dependence of ESS on the horizon | Not found | |
| Single-query ESS ≥ long-horizon ESS | Not found stated; proof is textbook (A- versus D-optimality) | Chaloner & Verdinelli 1995 |
| Long-horizon ESS equals information-equivalent examples | Known corollary | Clarke & Barron 1990 |
| Proportional-limit closed forms | Direct application | Tulino & Verdú 2004, eqs. 2.52, 2.61, 2.62 |
| Mis-set trust costs $\mathrm{KL}(p\,\|\,q)$ | Known corollary | Wrong-prior redundancy in universal coding |
| Reliability caps the worth of precision | Not found in this form | Reznik 2026, arXiv:2608.12159; Wiesenfarth & Calderazzo 2020 |
| Horizon reversal | Not found | |
| Networks match the Bayes-optimal learner | Published | Genewein et al. 2025; Zhu, Oermann & Cho 2026; Panwar et al. 2024 |

Papers to cite and distinguish: Reznik 2026 (power prior under predictive
log-loss; no mixture, no horizon dependence); Clarke 1996 and Lin, Pittman &
Clarke 2007 (information-theoretic prior sample size); Lu et al. 2025, PNAS
(random-matrix ICL error, no prior-to-examples conversion); Jeon et al. 2024
(cumulative log-loss, no single-query comparison); Tong et al. 2026 (a wrong
instruction washes out exponentially; no reliability parameter); Le Scao &
Rush 2021 and Cook et al. 2025 (empirical exchange rates in fine-tuning);
Majumdar 2023 (lower bounds when the prior may be wrong).

### Corrections to the campaign record

`docs/jobs/active/neurips-campaign/RESEARCH.md` describes two papers
inaccurately:

- Zhu, Oermann & Cho 2026 is "Multi-Task Bayesian In-Context Learning",
  arXiv:2606.20538. It is empirical, and its prefix is a set of datasets
  from earlier tasks, not a vector.
- Huang & Ge 2025: the descriptor is the mean of the inputs $x$. It says
  nothing about $w$.

Xuanyuan et al. 2025 could not be read beyond its abstract.

## Robustness of the reliability findings

| Finding in the campaign record | Result across four settings |
| --- | --- |
| Reliability caps the single-query worth of precision | Holds in all four |
| Horizon ordering reverses for precise, unreliable descriptions | Holds in three; fails at $d = 16$, `a0` $= 100$ |
| Trust errors are asymmetric | Mostly an artifact of trust exactly 1 |

Derived, and checked at one point: for a long horizon the ESS is the number
of examples whose information gain equals
$p\,\tfrac{d}{2}\log(1/r) - H(p)$, which is unbounded in precision. For a
single query the regret has a floor of about $(1-p)$ times the
no-description regret, so the worth saturates. The boundary of the reversal
is not yet derived.

## How comparable papers fared

- Accepted papers had a crisp, non-obvious message. Rejected ones were called
  incremental or unclear in their takeaway.
- The first version of Lin & Lee was rejected at ICLR 2024 (scores 3, 5, 6,
  6, 6) for a "more simplistic model than prior work" and "limited novelty"
  in Gaussian posteriors, with no champion. It was accepted at ICML 2024.
- Genewein et al. received a NeurIPS 2025 spotlight with toy tasks, small
  networks, and no LLM claims. The area chair called the findings
  "more-or-less already known" but valuable on a precise footing.
- Huang & Ge was accepted at ICLR 2025 (5, 6, 6, 8). Two of four reviewers
  asked why a mean vector is a good task description.
- Arora et al. had LLM experiments and was rejected at ICLR 2025.
- The toy setting was raised against accepted and rejected papers alike. An
  LLM experiment is not required in practice.

## Link to LLMs

No published work measures an instruction's worth in examples for an LLM.
Published data support two predictions qualitatively (a tipping point under
conflict; base models under-weighting instructions) and say nothing about
dimension or saturation. The cleanest free experiment is the coin-flip setup
of Gupta et al. 2025 with the stated bias and the flips in one prompt, on
open models of 8B or smaller. LLMs accumulate evidence sublinearly and tip
after tens of examples, where the exact learner tips after 1 to 3.

## Decisions

### Framing

- The instruction has two axes: precision and reliability. Dimension belongs
  to the task. Report ESS with $d$ as part of the setting; at high
  signal-to-noise the single-query ESS is about $(1-r)\,d$.
- Descriptions are statements about the mapping $w$. "Translate English to
  Spanish" is a coarse one; a stated rule is a precise one.
- Descriptions of the inputs (Huang & Ge) are worth zero examples to a
  Bayes-optimal learner, because the learner already sees each $x$. Their
  worth is computational: they help a limited model centre the inputs.

### RQ3 design, simplest version

| Piece | Before | Now | Reason |
| --- | --- | --- | --- |
| Precision | Four levels, stated as $\log r$ | One per model, not stated | Nothing for the model to misread; same treatment as reliability |
| Token | $3d + 4$ numbers | $d + 2$: vector, previous answer, flag | Matches Huang & Ge's prefix layout |
| Example token | $x_k$, $x_{k-1}$, $y_{k-1}$ | $x_k$, $y_{k-1}$ | The model saw $x_{k-1}$ one slot earlier |
| Slot marker | Marker and flag | Flag | Slot 0 is always the description |
| Output | Three Gaussian components | Two | The Bayes-optimal predictive has two |
| Architectures | Transformer and LSTM | Transformer first | LSTM is a later robustness check |

If training stalls, restore $x_{k-1}$ in the example token first.

### Deferred

- Runs at $d = 5$ to match Huang & Ge.
- Their input-mean descriptor as a second condition (informational versus
  computational worth).
- The LLM experiment.

## What the paper needs

1. One message: a description's worth depends on its reliability and on
   whether it can be checked against feedback.
2. The reversal boundary, derived.
3. A plain statement of which formulas are classical.
4. An early defence of the vector idealisation.
5. Network experiments that look for deviations from the Bayes-optimal
   learner, especially at test reliabilities different from training.

## Dates

| Venue | Deadline |
| --- | --- |
| NeurIPS 2027 | Not announced; early May in 2025 and 2026 |
| ICML 2027 | Late January 2027, unconfirmed |
| AISTATS 2027 | Paper 2026-10-06 |
| ICLR 2027 | Passed (2026-09-25) |
