# Literature

Closest work and how each differs. From the two novelty inquiries of
2026-09-29; papers marked † were read directly.

| Work | What it has | What it lacks |
| --- | --- | --- |
| Huang & Ge 2025 (ICLR) † | Descriptor token in linear regression; the descriptor is the input mean | The descriptor is independent of $w$, so a Bayes-optimal predictor values it at zero |
| Genewein et al. 2025 (NeurIPS) | Cumulative log-loss regret of small meta-trained networks against Bayes | No description, no ESS, no horizon comparison |
| Tong, Zeng & Zhang 2026 | A wrong instruction is overridden at an exponential rate | No reliability parameter, no ESS |
| Lin & Lee 2024 (ICML) | Gaussian-mixture task prior; early-ascent risk of a misleading component | No description, log loss, or ESS |
| Lin, Bharti & Lee 2025 | Hypothesis class as a prefix | Empirical only |
| Zhu, Oermann & Cho 2026 | Prefix of earlier datasets used Bayes-optimally | Empirical; prefix is data, not a vector |
| Lu et al. 2025 (PNAS) | Proportional-limit risk of linear attention | No prior-to-examples conversion |
| Jeon et al. 2024 (ICML) | Cumulative log loss by mutual information | No one-step comparison |
| Reimherr, Meng & Nicolae 2021 † | Prior sample size via discordance; names prediction-based ESS as open | The prediction-based version |
| Neuenschwander et al. 2020 † | Predictively consistent ESS, one parameter | Multi-parameter, prediction loss |
| Reznik 2026 (arXiv 2608.12159) † abstract | Power prior under predictive log loss; borrowed sample size capped by discrepancy | No mixture, no horizon dependence |
| Clarke & Barron 1990; Clarke 1996 | Regret equals information; information-theoretic prior sample size | The predictive matching |
| Chaloner & Verdinelli 1995 | A- versus D-optimality | |
| Le Scao & Rush 2021; Cook et al. 2025 | Empirical exchange rates in fine-tuning | A Bayes benchmark |

**Verdict of the inquiries.** The regret-matched ESS, its horizon
dependence, the reliability cap, and the reversal were found nowhere. The
long-horizon limit, the proportional-limit forms, and the KL cost are
classical. Not searched in depth: the minimum-description-length literature.

Corrections to the campaign record: Zhu et al. is empirical with a data
prefix; Huang & Ge's descriptor is the input mean. Xuanyuan et al. 2025 was
not read beyond its abstract.
