# Language models

**Status:** literature only; no experiment run or planned for the first
paper. Source: the LLM inquiry of 2026-09-29.

| Prediction | Published evidence | Status |
| --- | --- | --- |
| Crossover point when examples conflict with the instruction | Camassa & Shiller 2026 (arXiv 2605.20382); Bigelow et al. 2025 (arXiv 2511.00617) | Qualitatively supported, with anomalies |
| Lower effective reliability in models whose training text made instructions unreliable | Gupta et al. 2025 (ACL); Arora et al. 2024 | Qualitatively supported |
| One-step ESS grows with dimension | None | Untested |
| Saturation of precision when $p < 1$ | None | Untested |

**Discrepancies.** LLMs accumulate in-context evidence sublinearly (Bigelow
et al. fit $N^{1-\alpha}$), and their crossover points lie at tens of
demonstrations where the Bayes-optimal predictor's lie at one to three. Some
models never cross within 50 demonstrations; one crosses after one.

**Cheapest experiment.** Rerun the coin-flip setup of Gupta et al. locally
with the stated bias and the flips in one prompt; models of 8B or smaller
run on the Mac. Their paper never runs that joint condition.

**Caveat.** Most of these papers were read through page summaries; check
numbers against the PDFs before citing.
