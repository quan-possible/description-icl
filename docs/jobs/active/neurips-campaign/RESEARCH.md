# RESEARCH: what a task description is worth in examples

Task and continuity: [JOB.md](JOB.md). Last updated 2026-09-30.

## Question and standard

How many in-context examples is a task description worth, as a function of
its precision and reliability? The target is a NeurIPS 2027 paper, so the
result must change how the ICL community understands instructions, not only
be correct. Claims are limited to Bayes-optimal predictors and small
meta-trained models.

Notation: $\sigma_y = 1$; `a0` $= s_0^2/\sigma_y^2$; `r` $= s_\ell^2/s_0^2$;
$p$ is the true reliability and $q$ the learner's assumed reliability; $N$ is
the horizon.

## Assets

| Asset | Where |
| --- | --- |
| Exact learner, reliable descriptions | `src/descriptor_icl/gaussian.py` |
| Exact learner, unreliable descriptions | `src/descriptor_icl/mixture.py` |
| Meta-trained models | `src/descriptor_icl/meta.py` |
| RQ1 sweep | `results/rq1-single-query-gap/` |
| RQ2 tables | `results/rq2-reliability/` |
| RQ3 training | `results/rq3-meta-trained/` |

## What has been established

Status words: *derived* (follows from the model by argument), *computed*
(exact code, validated), *exploratory* (seen once, not yet in a committed
table).

1. **Everything for $p = 1$ reduces to one function** (derived). With
   $G_a(k) = \tfrac12 \mathbb{E}\log\det(I + a X_k^\top X_k)$, the
   description's regret over horizon $N$ is $G_{a_\ell}(N)$ and the regret
   after $n$ examples is $G_{a_0}(n+N) - G_{a_0}(n)$.
2. **Single-query ESS exceeds long-horizon ESS, by 1.01× to 8.5×**
   (computed, RQ1 grid). The ratio grows with $d$ and SNR and falls with
   precision. The proposal's "up to three times" is not a bound.
3. **Closed form in the proportional limit** (derived, checked against
   simulation at $d = 64$ and $d = 200$). Per-dimension ESS solves
   $\eta(\gamma) = r$ for a single query and $\mathcal{V}(\gamma) =
   \log(1/r)$ for the long horizon, with $\eta$ and $\mathcal{V}$ the
   Marchenko–Pastur trace and log-determinant transforms.
4. **Unreliable descriptions: regret splits into a Gaussian term and an
   identification term** (derived, validated). The identification term is
   the expected rise in the log posterior weight of the true component. Over
   a long horizon it tends to the cross-entropy $H(p, q)$, so mis-set trust
   costs exactly $\mathrm{KL}(p\,\|\,q)$ nats, independent of $d$.
5. **Reliability caps the single-query worth of precision** (computed, four
   settings, `results/rq2-reliability/ess_map.csv`; derived as the
   two-sided bound of Proposition 2 in the paper). At $d = 16$, `a0` $= 10$,
   $p = 0.9$, making the description 50 times more precise (`r` 0.05 →
   0.001) raises the single-query ESS from 14.8 to 23.1; at $p = 1$ it
   rises from 16.9 to 116.5.
6. **The horizon ordering reverses for precise, unreliable descriptions**
   (computed, three of four settings; `docs/wiki/horizon-reliability.md`).
   Not in the paper.
7. **Trust errors are asymmetric for a single query** (computed,
   `results/rq2-reliability/trust.csv`; `docs/wiki/misspecified-reliability.md`).
   Not in the paper.
8. **High-SNR ESS of a reliable description** (derived, Proposition 1):
   $\mathrm{ESS} = (d-1)(1-r) + O(1/(rd))$. Checked at `a0` $= 100$ to
   within 0.1.

## Hypotheses not supported

- I guessed that ESS might be non-monotone in precision when $p < 1$. In
  the exploratory table it is monotone. Precision saturates; it does not
  hurt a calibrated learner.

## Closest work (from a literature scan, 2026-09-29)

The scan was run by a subagent. Entries marked † were read in full text by
it; others rest on abstracts or snippets and need checking before citation.
I have not yet read any of them myself.

| Work | What it has | What it lacks relative to this project |
| --- | --- | --- |
| Tong et al. 2026 † | Examples overwhelm one wrong instruction exponentially fast (finite task set) | Exact counts, precision, mis-set trust, ESS |
| Zhu, Oermann & Cho 2026 † | Transformers use a prior-carrying prefix Bayes-optimally in linear regression | Descriptor vectors, reliability, ESS |
| Bigelow et al. 2025 † | Fitted crossover points between prior and examples in LLMs | Derivation, precision, instructions |
| Camassa & Shiller 2026 † | LLM instruction-versus-demonstration transition curves | A normative benchmark |
| Lin & Lee 2024 † | Gaussian-mixture task priors, closed-form posterior | Descriptor selects component; ESS |
| Reimherr et al. 2021 † | ESS depends on the discrepancy; prediction-based ESS left open | The prediction-based comparison itself |
| Neuenschwander et al. 2020 † | ESS for mixture priors, one parameter | Multi-parameter, prediction loss |
| Huang & Ge 2025 | Descriptor = input mean; training dynamics | Prior on $w$, ESS, reliability |
| Xuanyuan et al. 2025 | Weak descriptions hurt trained transformers | Bayes benchmark |

No work found prices an instruction in in-context examples. The inequality
in item 2 is AM–GM on posterior eigenvalues (A- versus D-optimality), so it
must be presented as a quantified mechanism, not a new inequality.

## Direction chosen (Bruce, 2026-09-29; the paper is the plan)

One-step ESS as a function of precision and reliability, then whether small
meta-trained Transformers reproduce it. Horizon dependence (A), the cost of
trust (B), and the LLM benchmark (E) are out of the paper and kept in
`docs/wiki/`. The closed form (C) is done for $p = 1$ in the proportional
limit and not used by the paper.

## Next moves

1. Bruce decides the network output and loss (squared error as coded, or log
   loss with a predicted spread, matching the paper's ESS definition).
2. Train in the paper's readout order (see `STATUS.md`, next actions) and
   fill section 6.
3. Standing adversarial review of the draft once section 6 has numbers.
