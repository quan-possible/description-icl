# Quantifying the Effective Sample Size of Task Descriptions in In-Context Learning

Bruce Quan Nguyen

*2026-10-01*

## Abstract

In-context learning (ICL) allows models to adapt to tasks using prompts that typically combine a natural language task description with input-output demonstrations. While the informational value of demonstrations is well understood, quantifying the precise contribution of the task description remains challenging. By formulating ICL as Bayesian inference over a latent task, we treat the description and demonstrations as distinct modalities of observations. We quantify the description’s value via its *effective sample size* (ESS)—the equivalent number of demonstrations required to achieve an identical reduction in one-step predictive log-loss regret for a Bayes-optimal predictor. In the exact setting of in-context linear regression, we demonstrate that a description’s ESS is governed by its *precision* (the reduction in prior variance) and its *reliability* (the probability of correct specification). We prove that for a correctly specified description, the ESS grows linearly in the task dimension $`d`$ when the description is coarse and linearly in $`1/r`$, the factor by which it shrinks the prior variance, when it is precise; the common form $`(d-1)(1-r) + 1/(r a_0)`$ is within two demonstrations of the exact ESS on our grid. Conversely, for a potentially misspecified description, the ESS is strictly bounded without demonstrations, as the optimal predictor must maintain posterior uncertainty over the description’s validity. Demonstrations act as evidence to resolve this unidentifiability, effectively unlocking the description’s value. Furthermore, we show that small meta-trained Transformers behave Bayes-optimally within the constraints of their output representations; when faced with misspecified descriptions, unimodal output heads force the model to project the optimal bimodal mixture onto a single Gaussian. Finally, we demonstrate that a network’s reliance on the description is inherited from its meta-training distribution, where full reliance on highly precise but misspecified descriptions yields more regret than ignoring the description.

## 1 Introduction

The empirical success of large language models is heavily reliant on in-context learning (ICL), where a model adapts to a novel task given a prompt (Brown et al. 2020). Standard prompts incorporate two distinct sources of task information: a natural language task description and a few demonstration examples. In practice the description is an instruction in natural language; we read it as a statement about the task, since its value lies in what it says the task is, and set aside how a model is made to follow it. Empirical observations regarding the description’s utility are heavily mixed; while a well-crafted description can substitute for hundreds of examples (Le Scao and Rush 2021; Honda et al. 2025), models are also known to perform adequately with misleading descriptions (Webson and Pavlick 2022) or heavily prioritize demonstrations over explicit instructions (Gupta et al. 2025). Existing theoretical frameworks for ICL often model the prompt strictly as a sequence of demonstrations (Garg et al. 2022; Akyürek et al. 2023; Xie et al. 2022), leaving a fundamental theoretical gap: how can we rigorously quantify the informational equivalent of a task description?

We address this by adopting a Bayesian perspective on ICL (Xie et al. 2022). In this framework, the pretraining distribution constitutes a prior over a latent task. The task description acts as a direct, explicit observation of the task parameters, whereas demonstrations serve as indirect observations. Conditioning on the description yields an updated task prior. In Bayesian statistics, the information contained in a prior can be quantified as its effective sample size (ESS) (Morita et al. 2008), which is dictated by the prior’s concentration (its *precision*) and the probability of prior-data conflict (its *reliability*) (Evans and Moshonov 2006).

We analyze this dynamic within noisy in-context linear regression with a Gaussian task prior (Garg et al. 2022; Akyürek et al. 2023; Raventós et al. 2023), where Bayes-optimal inference admits closed-form solutions. This tractability allows us to exactly compute a description’s ESS and provides exact reference predictors to evaluate the behavior of meta-trained Transformer models (Genewein et al. 2025).

## 2 Related Work

Our approach builds upon the understanding of ICL as implicit Bayesian inference (Xie et al. 2022), where meta-trained Transformers have been shown to implement optimal estimators such as Bayes-optimal ridge regression (Akyürek et al. 2023; Raventós et al. 2023; Panwar et al. 2024). While some literature has integrated task descriptions—modeling them as tokens that indicate hypothesis classes or carry input means (Huang and Ge 2025; Lin et al. 2025; Tong et al. 2026; Lin and Lee 2024)—these works generally do not quantify the description’s value in units of examples, nor do they formalize the probability of description misspecification. Closest to our setting, Zhu et al. (2026) show that a Transformer given a prefix of related datasets performs amortised hierarchical Bayesian prediction; we express the value of such conditioning information in demonstrations and allow it to be misspecified. Our ESS formulation aligns with prior sample size methodologies in Bayesian statistics (Morita et al. 2008; Reimherr et al. 2021; Neuenschwander et al. 2020), particularly those utilizing predictive regret (Reznik 2026). To model potential misspecification, we employ robust mixture priors, a technique commonly used in clinical trials to integrate potentially conflicting historical data (Schmidli et al. 2014).

## 3 Problem Formulation

#### Tasks and Demonstrations.

We assume a latent task represented by a weight vector $`w \in \mathbb{R}^d`$. A demonstration consists of an input $`x \sim \mathcal{N}(0, I_d)`$ and a label $`y = w^\top x + \epsilon`$, with observation noise $`\epsilon \sim \mathcal{N}(0, \sigma_y^2)`$. The base prior over tasks is isotropic Gaussian $`w \sim \mathcal{N}(0, s_0^2 I_d)`$. The prior signal-to-noise ratio (SNR) is defined as $`a_0 = s_0^2/\sigma_y^2`$, which we set to $`a_0 = 10`$ with $`\sigma_y = 1`$.

#### Task Descriptions.

A description is formalized as a vector $`m \in \mathbb{R}^d`$. Its generative process depends on a latent Bernoulli indicator $`Z \sim \text{Bernoulli}(p)`$, which dictates whether the description is correctly specified ($`Z=1`$) or misspecified ($`Z=0`$). The generative process is:
``` math
\begin{align}
    m &\sim \mathcal{N}(0, (1-r)s_0^2 I_d) \\
    w \mid m, Z=1 &\sim \mathcal{N}(m, r s_0^2 I_d) \\
    w \mid m, Z=0 &\sim \mathcal{N}(0, s_0^2 I_d)
\end{align}
```
Marginalizing over $`Z`$, the prior for $`w`$ given the description $`m`$ is the robust mixture $`p \mathcal{N}(m, rs_0^2 I_d) + (1-p) \mathcal{N}(0, s_0^2 I_d)`$ (Schmidli et al. 2014). The parameter $`r \in (0, 1)`$ sets the description’s **precision**: it is the ratio of the conditional variance to the base prior variance, so a smaller $`r`$ is a more precise description, and we plot $`r`$ with precision increasing to the right. The parameter $`p`$ denotes the description’s **reliability**. We call a description with $`p = 1`$ *correctly specified* and one with $`p < 1`$ *potentially misspecified*.

#### Regret and ESS.

We evaluate a predictor’s uncertainty given a prompt $`D`$ (the description, $`n`$ demonstrations, and a query $`x`$) via its one-step expected excess log-loss regret relative to an oracle that knows the true task $`w`$, the per-step regret of Genewein et al. (2025):
``` math
\begin{equation}
\label{eq:regret}
    R = \mathbb{E} \left[ -\log f(y \mid D) + \log \phi(y; w^\top x, \sigma_y^2) \right]
\end{equation}
```
where $`\phi`$ is the Gaussian density and $`f(y \mid D)`$ is the predictor’s predictive density. Let $`R^{\text{desc}}(n)`$ denote the Bayes-optimal regret with the description and $`n`$ demonstrations, $`R^{\text{ex}}(n)`$ the regret with $`n`$ demonstrations alone, and $`R^{\text{desc}}_{p=1}(n)`$ the former with a correctly specified description.

<div id="def:ess" class="definition">

**Definition 1**. *The Effective Sample Size (ESS) of a description, given $`n`$ demonstrations, is $`\text{ESS}_n = n^* - n`$, where $`n^*`$ is the root of $`R^{\text{ex}}(n^*) = R^{\text{desc}}(n)`$, calculated via linear interpolation. The standalone ESS is $`\text{ESS}_0`$.*

</div>

This is the data-dependent prior sample size of Reimherr et al. (2021) with predictive regret as the measure of uncertainty. It is negative when the description hurts, and when $`R^{\text{desc}}(n) \ge R^{\text{ex}}(0)`$ there is no root and we write $`\text{ESS}_n \le -n`$.

#### Reference Predictors and Reliance.

A learner applies an effective mixture weight $`q`$ to the description component. An *oracle-calibrated* predictor uses $`q=p`$ (Bayes-optimal). An *overconfident* predictor assumes $`q=1`$. We define a learner’s *implied reliance* as the value $`q`$ that minimizes the mean squared error between its posterior-mean predictions and those of the corresponding reference model.

## 4 The Value of Precision (Correctly Specified Descriptions)

When a description has perfect reliability ($`p=1`$), the posterior remains Gaussian. The regret is half the expected log of the remaining predictive variance, $`R = \frac{1}{2}\mathbb{E} \log(1 + x^\top \Sigma x)`$.

<div id="prop:reliable" class="proposition">

**Proposition 1**. *For $`p=1`$ and conjugate-prior ESS $`c = 1/(ra_0) = \sigma_y^2 / (rs_0^2)`$:*

1.  *(Coarse descriptions) If $`c \le 1`$, then as $`rd \to \infty`$, $`\text{ESS} = (d-1)(1-r) + (1-r)c + O(1/(rd))`$.*

2.  *(Precise descriptions) If $`c \ge d`$, then as $`c/d \to \infty`$, $`\text{ESS} = c + d + 1 - 1/a_0 + O(d/c)`$.*

3.  *For every $`d, a_0, r`$, we have $`\text{ESS} \le c + d + 2`$.*

</div>

Appendix [@.1](#app:precision) states the proposition in a fuller form and proves it. Both asymptotic regimes demonstrate that the ESS of a correctly specified description decomposes into a high-SNR term (representing the fraction of the dimension it constrains) and the conjugate-prior ESS $`c`$ (Neuenschwander et al. 2020). Roughly, $`\text{ESS} \approx (d-1)(1-r) + 1/(ra_0)`$; this is the two regimes’ common form, not a theorem for $`1 < c < d`$, where only the bound (3) applies, but on our grid it is within about two demonstrations of the exact ESS. At $`d = 16`$ and $`a_0 = 10`$ a description that removes nine tenths of the prior variance ($`r = 0.1`$) is worth $`14.9`$ demonstrations and one that leaves a thousandth ($`r = 0.001`$) is worth $`116`$ (Figure [1](#fig:ess)a).

## 5 The Identifiability Bottleneck (Potentially Misspecified Descriptions)

When $`p < 1`$, the Bayes-optimal posterior predictive is a two-component mixture. The optimal predictor must maintain uncertainty over the validity of the description.

<div id="prop:unreliable" class="proposition">

**Proposition 2**. Let $`\pi_n = P(Z = 1 \mid D_n)`$ be the posterior probability that the description is correctly specified given the prompt $`D_n`$, and let $`\pi_n(Z)`$ denote the posterior weight on the true mixture component: $`\pi_n`$ if $`Z = 1`$ and $`1 - \pi_n`$ otherwise. The regret is bounded by:
``` math
\begin{equation}
\label{eq:sandwich}
    p R^{\text{desc}}_{p=1}(n) + (1-p) R^{\text{ex}}(n) \le R^{\text{desc}}(n) \le p R^{\text{desc}}_{p=1}(n) + (1-p) R^{\text{ex}}(n) + \mathbb{E}[-\log \pi_n(Z)]
\end{equation}
```

</div>

The identification cost $`\mathbb{E}[-\log \pi_n(Z)]`$ is the conditional entropy $`H(Z \mid D_n)`$, which equals $`H(p)`$ at $`n = 0`$ and is non-increasing in $`n`$ (Appendix [@.2](#app:floor)). Without demonstrations ($`n=0`$), the regret is bounded below by $`(1-p)R^{\text{ex}}(0)`$. Consequently, the ESS strictly saturates at a finite value, preventing arbitrary scaling with precision as $`r \to 0`$. At $`d = 16`$, $`a_0 = 10`$ and $`p = 0.9`$ the limit is $`25`$ demonstrations (dotted in Figure [1](#fig:ess)c): refining $`r`$ from $`0.1`$ to $`0.001`$ raises the ESS of a correctly specified description from $`14.9`$ to $`116`$, but that of a $`p = 0.9`$ description only from $`13.1`$ to $`23.1`$.

<img src="../../results/rq2-reliability/ess.svg" />

**Figure 1.** ESS of a description against its precision $`r`$ with $`n`$ demonstrations in hand ($`\text{ESS}_n`$ of Definition [1](#def:ess)), one panel per reliability $`p`$; $`d = 16`$, $`a_0 = 10`$, one seed of 8,000 prompts. Dotted: the limit of the standalone ESS as $`r \to 0`$.

#### Demonstrations resolve unidentifiability.

As demonstrations accumulate, they provide evidence to infer the latent indicator $`Z`$, driving down the conditional entropy $`H(Z \mid D_n)`$. Heuristically, once the demonstrations have identified the description and $`R^{\text{ex}}(n) \approx d/(2n)`$, setting $`d/(2n^*)`$ equal to the lower bound of [\[eq:sandwich\]](#eq:sandwich) gives
``` math
\begin{equation}
\label{eq:ess-n}
\text{ESS}_n \;\approx\; \frac{n\,p\,c}{n + (1-p)\,c} \;\xrightarrow[n \to \infty]{}\; p\,c ,
\end{equation}
```
so the ESS of a potentially misspecified description approaches $`p`$ times that of a correctly specified one. This is an approximation, not a theorem, and the exact values bear it out: with $`100`$ demonstrations in hand, the $`r = 0.001`$, $`p = 0.9`$ description is worth $`88`$ demonstrations against $`82`$ from [\[eq:ess-n\]](#eq:ess-n) and $`108`$ for a correctly specified one (Figure [1](#fig:ess)). Demonstrations effectively pay down the informational cost of potential misspecification.

#### The cost of overconfidence.

An overconfident predictor ($`q=1`$) interacting with a misspecified description incurs severe regret. Because the assumed prior is highly concentrated around the erroneous description, the true generative target falls drastically outside the posterior predictive distribution. For highly precise descriptions, overconfidence is an active liability, yielding higher regret than ignoring the description entirely: at $`d = 16`$, $`a_0 = 10`$ and $`p = 0.9`$, the $`r = 0.001`$ description alone costs the overconfident predictor $`13.7`$ nats against $`2.51`$ for no description (closed forms; Figure [2](#fig:trust) shows the simulated curves). Under-reliance is cheap by comparison; the regret is flat for a wide band below $`p`$, rises gently above it, and jumps at $`q = 1`$.

![](../../results/rq2-reliability/trust.svg)

**Figure 2.** Regret of the Bayes-optimal predictor against the reliance $`q`$ it places on a description with reliability $`p = 0.9`$, alone (solid) and with ten demonstrations in hand (dashed); $`d = 16`$, $`a_0 = 10`$, one seed of 8,000 prompts.

## 6 Empirical Analysis: Meta-Trained Transformers

We meta-trained 12-layer Transformers on sequences generated by our problem formulation ($`d=5, a_0=10`$). The description $`m`$ is provided via a dedicated prefix token, optimized over Gaussian log-loss.

#### Setup.

Prompts use the prefix embedding of Huang and Ge (2025): an optional description token (their descriptor token) carrying $`m`$, up to $`20`$ demonstration tokens each holding an input beside its label, and a query token, with two indicator coordinates marking the token type and no positional encoding. The architecture and optimiser follow Garg et al. (2022): $`12`$ layers, $`8`$ heads, width $`256`$, Adam at a learning rate of $`10^{-4}`$, fresh prompts at every step, $`40{,}000`$ steps at batch $`1024`$ without a curriculum. The network outputs a mean and a log variance for the query’s label, so its regret is on the scale of [\[eq:regret\]](#eq:regret). The number of demonstrations is uniform on $`0`$ to $`20`$ and the description is omitted in half the prompts. We train one network for each combination of a precise ($`r = 0.01`$) or coarse ($`r = 0.1`$) description with reliability $`p \in \{1, 0.99, 0.9\}`$, one training run per setting and a second seed at $`p = 0.9`$, $`r = 0.01`$; the description token supplies only $`m`$, so each network must learn $`r`$ and $`p`$ from its meta-training distribution. We evaluate each network on $`20{,}000`$ fresh prompts for every $`n`$ from $`0`$ to $`20`$, with and without a description, on prompts of all three reliabilities. Its ESS follows Definition [1](#def:ess) with its own regret curves, and we place it between the reference predictors of §[3](#sec:setting) on the same prompts (Table [1](#tab:networks)).

<div id="tab:networks">

| $`r`$ | $`p_{\mathrm{train}}`$ | $`p_{\mathrm{test}}`$ | Network | Trained | Calibrated | Single Gaussian | Reliance | Description | None |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 1 | 1 | 5.24 $`\pm`$ 0.05 | 5.25 | — | — | 0.95 | 0.017 | 0.016 |
| 0.1 | 1 | 0.9 | 2.32 $`\pm`$ 0.14 | 2.05 | 4.26 | — | 0.95 | 0.010 | 0.016 |
| 0.01 | 1 | 1 | 14.74 $`\pm`$ 0.41 | 15.64 | — | — | 0.999 | 0.026 | 0.030 |
| 0.01 | 1 | 0.9 | $`\le 0`$ | $`\le 0`$ | 6.92 | — | 0.999 | -0.186 | 0.030 |
| 0.1 | 0.99 | 1 | 5.23 $`\pm`$ 0.05 | 5.26 | 5.25 | — | 0.95 | 0.018 | 0.018 |
| 0.1 | 0.99 | 0.99 | 5.01 $`\pm`$ 0.05 | 5.08 | — | 4.93 | 0.95 | 0.020 | 0.018 |
| 0.1 | 0.99 | 0.9 | 3.00 $`\pm`$ 0.09 | 3.97 | 4.26 | — | 0.95 | 0.046 | 0.018 |
| 0.01 | 0.99 | 1 | 11.82 $`\pm`$ 0.43 | 15.03 | 15.64 | — | 0.95 | 0.033 | 0.028 |
| 0.01 | 0.99 | 0.99 | 8.87 $`\pm`$ 0.22 | 12.88 | — | 8.54 | 0.95 | 0.042 | 0.028 |
| 0.01 | 0.99 | 0.9 | 0.07 $`\pm`$ 0.44 | 6.08 | 6.92 | — | 0.95 | 0.132 | 0.028 |
| 0.1 | 0.9 | 1 | 4.63 $`\pm`$ 0.04 | 5.05 | 5.25 | — | 0.85 | 0.025 | 0.018 |
| 0.1 | 0.9 | 0.9 | 3.57 $`\pm`$ 0.06 | 4.26 | — | 3.62 | 0.85 | 0.034 | 0.018 |
| 0.01 | 0.9 | 1 | 6.39 $`\pm`$ 0.06 | 11.90 | 15.64 | — | 0.85 | 0.055 | 0.022 |
| 0.01 | 0.9 | 0.9 | 4.30 $`\pm`$ 0.07 | 6.92 | — | 4.29 | 0.85 | 0.073 | 0.022 |
| 0.01 | 0.9 | 1 | 6.56 $`\pm`$ 0.06 | 11.90 | 15.64 | — | 0.9 | 0.057 | 0.024 |
| 0.01 | 0.9 | 0.99 | 6.27 $`\pm`$ 0.07 | 10.74 | 12.88 | — | 0.9 | 0.058 | 0.024 |
| 0.01 | 0.9 | 0.9 | 4.26 $`\pm`$ 0.08 | 6.92 | — | 4.29 | 0.9 | 0.078 | 0.024 |

**Table 1.** Networks against the reference predictors ($`d=5`$, $`a_0=10`$; 20,000 evaluation prompts per network, with the standard error of its ESS; the last three rows are the second seed). “Trained” is the Bayes-optimal predictor with reliance $`p_{\mathrm{train}}`$ and “calibrated” the one with reliance $`p_{\mathrm{test}}`$, computed once on 200,000 prompts (standard errors below $`0.12`$) and shown once where they coincide; “single Gaussian” is the best predictive a mean-and-variance head can express, where the Bayes-optimal predictive is a mixture. “Reliance” is the network’s implied reliance with the description alone, on a grid of step $`0.05`$ plus $`0.99`$ and $`0.999`$. The last two columns are the mean regret gap, network minus the trained predictor, over $`n = 0, \dots, 20`$, with and without a description.

</div>

#### Expressivity bounds Bayes-optimal behavior.

On their respective training distributions, Transformers approach the Bayes-optimal ESS for perfectly specified descriptions: $`14.7 \pm 0.4`$ against $`15.6`$ at $`r = 0.01`$ and $`5.24 \pm 0.05`$ against $`5.25`$ at $`r = 0.1`$. However, for potentially misspecified descriptions with zero demonstrations, the networks fall short of the exact Bayes-optimal mixture ESS: at $`p = 0.9`$, $`4.30 \pm 0.07`$ against $`6.92`$ and $`3.57 \pm 0.06`$ against $`4.26`$, and at $`p = 0.99`$, $`8.9 \pm 0.2`$ against $`12.9`$ and $`5.01 \pm 0.05`$ against $`5.08`$. Standard regression heads emit a single mean and variance, inherently lacking the capacity to represent a bimodal posterior mixture. Consequently, the networks project the optimal mixture onto the *best single Gaussian* via moment matching, whose ESS is $`4.29`$, $`3.62`$, $`8.54`$, and $`4.93`$ in the same four cases. As demonstrations collapse the posterior mixture to a single component, the networks successfully converge with the exact Bayes-optimal predictor: at $`r = 0.01`$, $`p = 0.9`$ the network’s regret is within $`0.05`$ nats of it from $`n = 6`$ on, and its ESS with ten demonstrations in hand is $`8.1`$ against $`9.0`$. A second seed reproduces the first ($`4.26 \pm 0.08`$ alone, $`8.3`$ with ten demonstrations). The shortfall is not under-training: a model trained at $`r = 0.05`$, $`p = 0.9`$ and continued to $`80{,}000`$ steps kept it ($`3.9`$ against $`5.0`$).

#### Reliance is inherited from meta-training.

We demonstrate that the implied reliance of the network is governed by the meta-training distribution. Models trained exclusively on perfect reliability ($`p=1`$) learn an implied reliance of $`q \approx 1`$. At deployment, they act as overconfident predictors: when one description in ten is misspecified, the precise model’s regret with the description alone is $`2.87`$ nats against $`1.87`$ without it (and $`3.32`$ for the overconfident reference, which it beats by hedging slightly, at an implied reliance of $`0.999`$), so its ESS is at most zero where the oracle-calibrated predictor obtains $`6.9`$. Models trained at $`p = 0.99`$ read a reliance of $`0.95`$ and lose the precise description’s entire value when one in ten is misspecified ($`0.1 \pm 0.4`$ against $`6.9`$). Conversely, models trained on $`p=0.9`$ learn to maintain structural uncertainty, exhibiting an implied reliance of $`0.85`$ to $`0.9`$ across the two seeds even when evaluated exclusively on correctly specified descriptions, where their ESS is $`6.4`$ and $`6.6`$ against the oracle-calibrated $`15.6`$ at $`r = 0.01`$.

## 7 Discussion and Limitations

Our theoretical framework establishes that the ESS of a valid task description scales with its precision and constrained dimensionality. However, potential misspecification introduces an identifiability bottleneck, forcing optimal predictors to maintain uncertainty and capping the description’s value. Demonstrations are critical not just as additive data, but as verification signals that resolve this unidentifiability. Meta-trained Transformers approximate this Bayes-optimal reasoning, bounded by the expressivity of their unimodal output representations, with their propensity for reliance inherited from training.

Our analysis is subject to limitations, notably the assumption of a linear-Gaussian generative process; whether our claims carry over to large language models reading text is open, and the network results rest on one training run per setting, two for one of them. Furthermore, our ESS metric is defined via one-step prediction; under sequential prediction horizons where the predictor’s own feedback accumulates over time, the ESS is lower ($`7.9`$ against $`14.9`$ demonstrations over a horizon of $`200`$ predictions at $`d = 16`$, $`a_0 = 10`$, $`r = 0.1`$), representing a valuable direction for future research.

## 8 Mathematical Proofs

### Proof of Proposition [1](#prop:reliable) (Correctly Specified Descriptions)

We prove Proposition [1](#prop:reliable) using the exact regret expansions for the predictive log-loss. Let $`\psi`$ denote the digamma function, and let $`\varepsilon = 1/a_0`$. Given $`n`$ demonstrations $`X_n \in \mathbb{R}^{n \times d}`$, the posterior covariance is $`\Sigma = \sigma_y^2 (\varepsilon I + X_n^\top X_n)^{-1}`$. For a query $`x \sim \mathcal{N}(0, I_d)`$, let $`A = \|x\|^2 \sim \chi^2_d`$. The one-step expected regret is $`R^{\text{ex}}(n) = \frac{1}{2} \mathbb{E} \log(1 + x^\top \Sigma x)`$.

#### Regret Expansions.

Using properties of Wishart distributions and the Sherman-Morrison formula, we can bound the regret in two regimes:

1.  **Fewer than $`d`$ demonstrations ($`n < d`$):** Let $`k = d - n`$. The regret is lower bounded by:
    ``` math
    \begin{equation}
            2R^{\text{ex}}(n) \ge \log(2a_0) + \psi(k/2)
        
    \end{equation}
    ```
    with equality as $`a_0 \to \infty`$.

2.  **At least $`d`$ demonstrations ($`n \ge d`$):** Let $`\ell = n - d + 1`$. The regret is upper bounded by:
    ``` math
    \begin{equation}
            2R^{\text{ex}}(n) \le \psi\left(\frac{n+1}{2}\right) - \psi\left(\frac{\ell}{2}\right)
        
    \end{equation}
    ```

#### Locating the ESS Root.

By Definition 1, the ESS is the root $`n^*`$ of $`R^{\text{ex}}(n^*) = R^{\text{desc}}(0)`$. For a correctly specified description, $`R^{\text{desc}}(0) = \frac{1}{2}\mathbb{E}\log(1 + c^{-1} A)`$ where $`c = 1/(ra_0)`$. Matching the regret expansions to $`R^{\text{desc}}(0)`$ and applying the asymptotic expansions of the digamma function ($`\psi(z) = \log z - 1/(2z) + O(z^{-2})`$) yields the roots:

- In the coarse regime ($`c \le 1`$), matching the lower bound gives $`d - n \approx rd`$, leading to $`\text{ESS} \approx (d-1)(1-r) + (1-r)c`$.

- In the precise regime ($`c \ge d`$), matching the upper bound gives $`n \approx c + d + 1`$, leading to $`\text{ESS} \approx c + d + 1 - 1/a_0`$.

By concavity of the log-loss, the ESS is at most $`c + d + 2`$ in every regime.

The remainder of this section makes the sketch rigorous: it states the proposition in a fuller form, with the regret expansions behind (1) and (2) and their error terms, proves the two bounds above as lemmas with their expansions, and locates the root.

**Proposition [1](#prop:reliable) (full statement).**

*Let $`p = 1`$, let $`\psi`$ be the digamma function, and let $`c = 1/(r a_0) = \sigma_y^2 / s_1^2`$ with $`s_1^2 = r s_0^2`$: the ESS of the description’s prior $`\mathcal{N}(\ell, s_1^2 I_d)`$ in the conjugate sense, noise variance over prior variance (Neuenschwander et al. 2020), the conjugate-prior ESS.*

1.  *Coarse descriptions.* Suppose $`c \le 1`$. Then for every $`n < d`$, $`R^{\mathrm{ex}}(n) \ge \tfrac12\big[\log(2a_0) + \psi\big(\tfrac{d-n}{2}\big)\big]`$, with equality in the limit $`a_0 \to \infty`$, and as $`rd \to \infty`$, uniformly in $`c \le 1`$,
    ``` math
    \mathrm{ESS}= (d-1)(1-r) + (1-r)\,c + O\!\left(\tfrac{1}{rd}\right).
    ```

2.  *Precise descriptions.* Suppose $`c \ge d`$. Then for every $`n \ge d`$, $`R^{\mathrm{ex}}(n) \le \Phi(n) := \tfrac12\big[\psi\big(\tfrac{n+1}{2}\big) - \psi\big(\tfrac{n-d+1}{2}\big)\big]`$, with equality in the limit $`a_0 \to \infty`$, and as $`c/d \to \infty`$, uniformly in $`a_0 \ge 1`$,
    ``` math
    \mathrm{ESS}= c + d + 1 - \tfrac{1}{a_0} + O\!\left(\tfrac{d}{c}\right).
    ```

3.  For every $`d`$, $`a_0`$, and $`r`$, $`\mathrm{ESS}\le c + d + 2`$.

#### Standard facts.

Throughout, $`\varepsilon = 1/a_0`$, $`A = \|x\|^2 \sim \chi^2_d`$ for the query $`x`$, $`\Sigma = (\varepsilon I + X_n^\top X_n)^{-1}`$, and $`g(t) = \mathbb{E}\log(1 + At)`$, which is increasing and concave with
``` math
\begin{equation}
\label{eq:g-derivs}
d\,(1 - (d+2)t) \le g'(t) = \mathbb{E}\frac{A}{1 + At} \le d,
\qquad
-(d^2 + 2d) \le g''(t) \le 0 .
\end{equation}
```
We use (a) $`\mathbb{E}\log \chi^2_k = \psi(k/2) + \log 2`$, $`\mathbb{E}[\chi_k^{-2j}] = \prod_{i=1}^{j} (k - 2i)^{-1}`$ for $`k > 2j`$, and $`\psi(z) = \log z - 1/(2z) + O(z^{-2})`$; (b) for $`W \sim W_j(\nu, I)`$ with $`\nu \ge j`$ and $`z \sim \mathcal{N}(0, I_j)`$ independent of $`W`$, $`z^\top W^{-1} z`$ is distributed as $`\chi^2_j / \chi^2_{\nu-j+1}`$ with independent numerator and denominator (Muirhead 1982, Theorem 3.2.12), and $`\mathbb{E}\operatorname{tr} W^{-2} = j(\nu-1)/((\nu-j)(\nu-j-1)(\nu-j-3))`$ for $`\nu - j \ge 4`$ (Rosen 1988); (c) $`y - y^2/2 \le \log(1 + y) \le y`$ for $`y \ge 0`$.

#### Regret.

The posterior predictive is the true conditional law of $`y`$, a Gaussian with variance $`1 + x^\top \Sigma x`$, so its expected log loss is its entropy and $`R^{\mathrm{ex}}(n) = \tfrac12 \mathbb{E}\log(1 + x^\top \Sigma x)`$, as in §[4](#sec:precision). It is strictly decreasing in $`n`$, since by the Sherman–Morrison formula each new demonstration lowers $`x^\top \Sigma x`$ almost surely. For the description, $`n = 0`$ and $`\Sigma = r a_0 I = I/c`$, so by (a) and (c) with $`y = c/A`$,
``` math
\begin{equation}
\label{eq:rdesc}
2R^{\mathrm{desc}} = g(1/c) = \log(2 r a_0) + \psi(d/2) + \delta^{\mathrm{desc}},
\qquad
\delta^{\mathrm{desc}} = \mathbb{E}\log(1 + c/A) = \frac{c}{d-2} + O\!\left(\frac{c^2}{d^2}\right) \quad (d \ge 5).
\end{equation}
```

#### Locating the root.

Let $`h(n) = 2R^{\mathrm{ex}}(n) - 2R^{\mathrm{desc}}`$ on integers $`n \ge 0`$, with linear interpolant $`\bar h`$. Then $`\mathrm{ESS}`$ is the unique root of $`\bar h`$, and $`h(0) = g(a_0) - g(r a_0) > 0`$. In (i) and (ii) we show, for some $`n_0 \ge 0`$, $`\alpha > 0`$, and $`\theta \to 0`$ along the stated limit,
``` math
\begin{equation}
\label{eq:root}
h(n) = \alpha\,(n_0 - n) + O(\alpha\theta)
\qquad \text{uniformly over integers $n \ge 0$ with $|n - n_0| \le 3$.}
\end{equation}
```
Being a convex combination of two such values, $`\bar h(t)`$ obeys the same expansion at real $`t \ge 0`$ with $`|t - n_0| \le 2`$; so for small $`\theta`$ the root lies within $`2`$ of $`n_0`$, and $`0 = \bar h(\mathrm{ESS}) = \alpha(n_0 - \mathrm{ESS}) + O(\alpha\theta)`$ gives $`\mathrm{ESS}= n_0 + O(\theta)`$.

<div id="lem:few" class="lemma">

**Lemma 1** (fewer than $`d`$ demonstrations). Let $`n < d`$ and $`k = d - n`$. Then $`2R^{\mathrm{ex}}(n) = \log(2a_0) + \psi(k/2) + \delta^{\mathrm{ex}}`$ with $`\delta^{\mathrm{ex}} \ge 0`$, $`\delta^{\mathrm{ex}} \to 0`$ as $`a_0 \to \infty`$, and, for $`k \ge 5`$,
``` math
\delta^{\mathrm{ex}} = \frac{\varepsilon (d-1)}{(k-1)(k-2)} + O\!\left(\frac{\varepsilon^2 d^2}{k^4}\right).
```

</div>

<div class="proof">

*Proof.* Write $`X_n^\top X_n = \sum_{i \le n} \lambda_i u_i u_i^\top`$ with $`\lambda_i > 0`$ almost surely, and let $`Q`$ be the squared norm of the projection of $`x`$ onto the $`k`$ directions orthogonal to the $`u_i`$. Then $`\Sigma`$ has eigenvalue $`a_0`$ on those directions and $`1/(\varepsilon + \lambda_i)`$ on $`u_i`$, so
``` math
1 + x^\top \Sigma x = a_0 Q + T, \qquad T = 1 + V, \quad V = \sum_{i \le n} \frac{z_i^2}{\varepsilon + \lambda_i}, \quad z_i = u_i^\top x,
```
where, given $`X_n`$, the $`z_i`$ are i.i.d. standard normal and independent of $`Q \sim \chi^2_k`$. Taking logarithms and expectations, by (a),
``` math
2R^{\mathrm{ex}}(n) = \log(2a_0) + \psi(k/2) + \delta^{\mathrm{ex}}, \qquad \delta^{\mathrm{ex}} = \mathbb{E}\log\!\left(1 + \frac{\varepsilon T}{Q}\right) \ge 0 .
```
As $`a_0 \to \infty`$, $`\varepsilon T`$ decreases to $`0`$ pathwise, so $`\delta^{\mathrm{ex}} \to 0`$ by dominated convergence (the integrand at $`a_0 = 1`$ is integrable).

For the expansion, let $`F = \sum_i z_i^2 / \lambda_i`$. In the eigenbasis of $`W = X_n X_n^\top \sim W_n(d, I)`$, which has the same eigenvalues, $`F = z^\top W^{-1} z`$ and $`\sum_i z_i^2/\lambda_i^2 = z^\top W^{-2} z`$ with $`z \sim \mathcal{N}(0, I_n)`$ independent of $`W`$. By (b), $`F \sim \chi^2_n / \chi^2_{k+1}`$, so $`\mathbb{E}F = n/(k-1)`$ and $`\mathbb{E}(1+F)^2 = O(d^2/k^2)`$, and $`\mathbb{E}\sum_i z_i^2/\lambda_i^2 = n(d-1)/(k(k-1)(k-3))`$. Since $`1/\lambda - \varepsilon/\lambda^2 \le 1/(\varepsilon + \lambda) \le 1/\lambda`$, we have $`F - \varepsilon \sum_i z_i^2/\lambda_i^2 \le V \le F`$. Now (c) with $`y = \varepsilon T / Q`$, the independence of $`Q`$ from $`T`$, and (a) give
``` math
\frac{\varepsilon\,(1 + \mathbb{E}V)}{k-2} - \frac{\varepsilon^2\, \mathbb{E}T^2}{2(k-2)(k-4)} \le \delta^{\mathrm{ex}} \le \frac{\varepsilon\,(1 + \mathbb{E}F)}{k-2} = \frac{\varepsilon (d-1)}{(k-1)(k-2)} ,
```
and the two sides differ by $`\varepsilon^2 \mathbb{E}\sum_i z_i^2/\lambda_i^2 /(k-2) + \varepsilon^2 \mathbb{E}T^2 / (2(k-2)(k-4)) = O(\varepsilon^2 d^2 / k^4)`$, as $`\mathbb{E}T^2 \le \mathbb{E}(1 + F)^2`$. ◻

</div>

<div id="lem:many" class="lemma">

**Lemma 2** (at least $`d`$ demonstrations). Let $`n \ge d`$ and $`\ell = n - d + 1`$. Then $`2\Phi(n) = \mathbb{E}\log(1 + A/Q)`$ with $`Q \sim \chi^2_\ell`$ independent of $`A`$, and $`R^{\mathrm{ex}}(n) \le \Phi(n)`$, with equality as $`a_0 \to \infty`$. If moreover $`a_0 \ge 1`$ and $`\ell \ge \max(d, 7)`$, then
``` math
2R^{\mathrm{ex}}(n) = 2\Phi(n) - \frac{\varepsilon d}{\ell^2} + O\!\left(\frac{\varepsilon d^2}{\ell^3}\right).
```

</div>

<div class="proof">

*Proof.* Rotate coordinates so that $`x = \|x\| e_1`$; the rows of $`X_n`$ remain i.i.d. $`\mathcal{N}(0, I_d)`$, independent of $`A`$. Then $`x^\top \Sigma x = A\,\Sigma_{11} = A/S`$ with $`S`$ the Schur complement of the first coordinate in $`\Sigma^{-1}`$, which the Woodbury identity writes, with $`X_1`$ the first column of $`X_n`$ and $`X_{-1}`$ the other $`d-1`$ columns, as
``` math
S = \varepsilon + X_1^\top \big(I_n + a_0 X_{-1} X_{-1}^\top\big)^{-1} X_1 .
```
Given $`X_{-1}`$, the middle matrix has eigenvalue $`1`$ on the $`\ell`$ directions orthogonal to the columns of $`X_{-1}`$ and $`1/(1 + a_0 \mu_j)`$ on the other $`d - 1`$, where $`\mu_j`$ are the eigenvalues of $`M = X_{-1}^\top X_{-1} \sim W_{d-1}(n, I)`$. Since $`X_1 \sim \mathcal{N}(0, I_n)`$ is independent of $`X_{-1}`$, its coordinates in that eigenbasis are i.i.d. standard normal, and
``` math
S = Q + \Delta, \qquad Q \sim \chi^2_\ell, \qquad
\Delta = \varepsilon + \sum_{j < d} \frac{z_j^2}{1 + a_0 \mu_j} \in \big[\varepsilon,\, \varepsilon(1 + F)\big], \qquad F = \sum_{j < d} \frac{z_j^2}{\mu_j},
```
with $`A`$, $`Q`$, and $`(z, M)`$ independent; by (b), $`F \sim \chi^2_{d-1} / \chi^2_{\ell+1}`$, so $`\mathbb{E}F = (d-1)/(\ell-1)`$ and, when $`\ell \ge \max(d, 7)`$, $`\mathbb{E}(1+F)^2 \le 5`$.

Since $`A + Q \sim \chi^2_{n+1}`$, (a) gives $`\mathbb{E}\log(1 + A/Q) = \psi\big(\tfrac{n+1}{2}\big) - \psi\big(\tfrac{\ell}{2}\big) = 2\Phi(n)`$. As $`S \ge Q`$, $`R^{\mathrm{ex}}(n) \le \Phi(n)`$, with equality as $`a_0 \to \infty`$ by monotone convergence, since $`\Delta`$ decreases to $`0`$.

For the expansion, $`2R^{\mathrm{ex}}(n) = 2\Phi(n) - \mathbb{E}D`$ with
``` math
D = \log\Big(1 + \frac{A}{Q}\Big) - \log\Big(1 + \frac{A}{S}\Big) = \log(1 + y_1) - \log(1 + y_2), \quad y_1 = \frac{\Delta}{Q} \ge y_2 = \frac{\Delta}{Q + A}, \quad y_1 - y_2 = \frac{\Delta A}{Q(Q+A)} .
```
By concavity of $`\log(1+y)`$ and $`1/(1+y_1) \ge 1 - y_1`$, $`(y_1 - y_2)(1 - \Delta/Q) \le D \le y_1 - y_2`$. Here $`\Delta`$ is independent of $`(A, Q)`$; $`\mathbb{E}\frac{A}{Q(Q+A)} = \mathbb{E}\frac1Q - \mathbb{E}\frac{1}{Q+A} = \frac{d}{(\ell-2)(\ell+d-2)}`$ by (a), as $`Q + A \sim \chi^2_{\ell+d}`$; $`\frac{A}{Q^2(Q+A)} \le \frac{A}{Q^3}`$; and $`\varepsilon \le \mathbb{E}\Delta \le \varepsilon(1 + \mathbb{E}F) = \varepsilon\frac{\ell+d-2}{\ell-1}`$, $`\mathbb{E}\Delta^2 \le 5\varepsilon^2`$. Hence
``` math
\frac{\varepsilon d}{(\ell-2)(\ell+d-2)} - \frac{5\,\varepsilon^2 d}{(\ell-2)(\ell-4)(\ell-6)} \le \mathbb{E}D \le \frac{\varepsilon d}{(\ell-1)(\ell-2)} ,
```
and both sides are $`\varepsilon d / \ell^2 + O(\varepsilon d^2 / \ell^3)`$, using $`\varepsilon \le 1`$. ◻

</div>

<div class="proof">

*Proof of Proposition [1](#prop:reliable).* *(iii).* Conditionally on $`A`$, $`\log(1 + Av)`$ is concave in $`v`$, so Jensen’s inequality with $`\mathbb{E}[1/Q] = 1/(\ell-2)`$ gives $`2\Phi(n) \le g(1/(\ell-2))`$ for $`\ell \ge 3`$. If $`\ell - 2 \ge c`$, that is $`n \ge c + d + 1`$, then $`g(1/(\ell-2)) \le g(1/c) = 2R^{\mathrm{desc}}`$, so $`R^{\mathrm{ex}}(n) \le \Phi(n) \le R^{\mathrm{desc}}`$ by Lemma [2](#lem:many). The least such integer is $`\lceil c + d + 1 \rceil < c + d + 2`$, and $`R^{\mathrm{ex}}`$ is decreasing, so $`\mathrm{ESS}\le c + d + 2`$.

*(i).* The lower bound and the limit are Lemma [1](#lem:few). For the expansion let $`c \le 1`$, $`n_0 = (d-1)(1-r) + (1-r)c`$, and $`n \ge 0`$ an integer with $`|n - n_0| \le 3`$; write $`k = d - n = rd + u`$. Since $`n_0 = d - rd - u_0`$ with $`u_0 = 1 - r + \varepsilon - c \in [-c, 1]`$, we have $`|u| \le 4`$, so $`k \ge 5`$ once $`rd \ge 9`$. By Lemma [1](#lem:few), [\[eq:rdesc\]](#eq:rdesc), $`\psi(z) = \log z - 1/(2z) + O(z^{-2})`$, and $`\varepsilon = rc`$, with every $`O`$ uniform in $`c \le 1`$ and $`|u| \le 4`$,
``` math
\begin{align*}
h(n) &= \psi(k/2) - \psi(d/2) - \log r + \delta^{\mathrm{ex}} - \delta^{\mathrm{desc}} \\
&= \Big[\log\frac{k}{rd} - \frac1k + \frac1d\Big] + \frac{\varepsilon d}{r^2 d^2}\Big(1 + O\Big(\frac{1}{rd}\Big)\Big) - \frac{c}{d} + O\Big(\frac{1}{(rd)^2}\Big)
 = \frac{u - 1 + r + c - \varepsilon}{rd} + O\Big(\frac{1}{(rd)^2}\Big),
\end{align*}
```
since $`\log(k/(rd)) = u/(rd) + O((rd)^{-2})`$, $`1/k = 1/(rd) + O((rd)^{-2})`$, $`\varepsilon d / (r^2 d^2) = c/(rd)`$, and $`c/d = \varepsilon/(rd)`$. So $`h(n) = \frac{1}{rd}\big[(n_0 - n) + O(1/(rd))\big]`$, which is [\[eq:root\]](#eq:root) with $`\alpha = \theta = 1/(rd)`$, and $`\mathrm{ESS}= (d-1)(1-r) + (1-r)c + O(1/(rd))`$.

*(ii).* The upper bound and the limit are Lemma [2](#lem:many). For the expansion let $`n_0 = c + d + 1 - \varepsilon`$ and $`n`$ an integer with $`|n - n_0| \le 3`$, so that $`\ell - 2 = c + s`$ with $`s = n - n_0 - \varepsilon`$, $`|s| \le 4`$; for $`c \ge 9d`$ every such $`n`$ has $`\ell \ge \max(d, 7)`$. By [\[eq:rdesc\]](#eq:rdesc) and Lemma [2](#lem:many), $`2\Phi(n) - 2R^{\mathrm{desc}} = \mathbb{E}\big[g(1/Q) - g(1/c)\big]`$, and a second-order Taylor expansion of $`g`$ about $`1/c`$ with [\[eq:g-derivs\]](#eq:g-derivs), $`\mathbb{E}[1/Q] - 1/c = -s/(c(c+s))`$, and $`\mathbb{E}(1/Q - 1/c)^2 = \operatorname{Var}(1/Q) + s^2/(c(c+s))^2 = O(c^{-3})`$ gives
``` math
2\Phi(n) - 2R^{\mathrm{desc}} = -\frac{s}{c(c+s)}\,d\,\Big(1 + O\Big(\frac dc\Big)\Big) + O\Big(\frac{d^2}{c^3}\Big) = -\frac{ds}{c^2} + O\Big(\frac{d^2}{c^3}\Big).
```
With Lemma [2](#lem:many), $`\varepsilon \le 1`$, and $`\varepsilon d/\ell^2 = \varepsilon d/c^2 + O(d/c^3)`$,
``` math
h(n) = -\frac{d}{c^2}\Big[s + \varepsilon + O\Big(\frac{d}{c}\Big)\Big] = \frac{d}{c^2}\Big[(n_0 - n) + O\Big(\frac{d}{c}\Big)\Big],
```
which is [\[eq:root\]](#eq:root) with $`\alpha = d/c^2`$ and $`\theta = d/c`$, uniformly in $`a_0 \ge 1`$, so $`\mathrm{ESS}= c + d + 1 - 1/a_0 + O(d/c)`$. ◻

</div>

### Proof of Proposition [2](#prop:unreliable) (Potentially Misspecified Descriptions)

Let $`D_n = (x_{1:n}, y_{1:n}, m, x)`$ denote the prompt and the query input. Let $`Z \in \{0, 1\}`$ be the true validity indicator of the description.

#### Step 1: The Mixture Predictive.

The Bayes-optimal posterior predictive is a $`\pi_n`$-mixture of the correctly specified predictive $`f_1`$ and the base prior predictive $`f_0`$, where $`\pi_n = P(Z=1 \mid D_n)`$. That is, $`f(y) = \pi_n f_1(y) + (1 - \pi_n) f_0(y)`$, where $`f_1`$ and $`f_0`$ are the predictives of the two components given $`D_n`$: $`f_1`$ is the correctly specified predictor’s, and $`f_0`$ is the demonstrations-only predictor’s, because under $`Z = 0`$ the description is independent of $`w`$ and carries no information about $`y`$. At $`n = 0`$, $`\pi_0 = p`$: seeing $`m`$ says nothing about $`Z`$, because $`m`$ has the same law $`\mathcal{N}(0, (1-r) a_0 I)`$ either way, and $`x`$ is independent of $`Z`$, so there is nothing to update on. The components are then $`f_1 = \mathcal{N}(m^\top x,\, 1 + r a_0\|x\|^2)`$ and $`f_0 = \mathcal{N}(0,\, 1 + a_0\|x\|^2)`$.

#### Step 2: The Oracle Twin Lower Bound.

Consider an oracle predictor that is explicitly given $`Z`$. This twin predicts using $`f_1`$ when $`Z=1`$ (incurring regret $`R^{\text{desc}}_{p=1}(n)`$) and $`f_0`$ when $`Z=0`$ (incurring regret $`R^{\text{ex}}(n)`$). The twin’s expected regret is exactly $`p R^{\text{desc}}_{p=1}(n) + (1-p) R^{\text{ex}}(n)`$. Because additional information cannot hurt a Bayes-optimal predictor, the expected log-loss of our mixture predictor $`f`$ cannot be lower than the oracle twin’s log-loss. Explicitly, given $`(D_n, Z)`$ the true conditional density of $`y`$ is $`f_Z`$, and the expected log loss of any density $`q`$ exceeds that of $`f_Z`$ by $`\mathbb{E}[-\log q \mid D_n, Z] - \mathbb{E}[-\log f_Z \mid D_n, Z] = \mathrm{KL}(f_Z \,\|\, q) \ge 0`$; averaging over $`Z`$ and $`D_n`$ gives the bound. Thus:
``` math
\begin{equation}
    R^{\text{desc}}(n) \ge p R^{\text{desc}}_{p=1}(n) + (1-p) R^{\text{ex}}(n)
\end{equation}
```

#### Step 3: The Identification Cost Upper Bound.

Pointwise, the mixture density satisfies $`f \ge \pi_n f_1`$ and $`f \ge (1-\pi_n) f_0`$. Consequently:
``` math
\begin{align}
    -\log f &\le -\log f_1 - \log \pi_n \\
    -\log f &\le -\log f_0 - \log(1-\pi_n)
\end{align}
```
Taking the expectation over the true data generating process $`f_Z`$ and the latent variable $`Z`$ yields the upper bound. The penalty term simplifies to the conditional entropy $`\mathbb{E}[-\log \pi_n(Z)] = H(Z \mid D_n)`$. At $`n = 0`$ it is $`p \log(1/p) + (1-p)\log(1/(1-p)) = H(p)`$.

#### Step 4: Monotonicity and Saturation.

The conditional entropy $`H(Z \mid D_n)`$ is non-increasing in $`n`$, since conditioning on additional demonstrations can only reduce entropy. At $`n=0`$, $`H(Z \mid D_0) = H(p)`$. In detail, the query $`x`$ is independent of $`Z`$ and of everything else in $`D_n`$, so $`H(Z \mid D_n) = H(Z \mid x_{1:n}, y_{1:n}, m)`$, and these conditioning sets are nested in $`n`$. Evaluating the lower bound at $`n=0`$ gives $`R^{\text{desc}}(0) \ge (1-p)R^{\text{ex}}(0)`$. Since $`R^{\text{ex}}(n)`$ is strictly decreasing in $`n`$, there exists a finite $`n^*`$ such that $`R^{\text{ex}}(n^*) = (1-p)R^{\text{ex}}(0)`$. The ESS cannot exceed this $`n^*`$, so it saturates. The lower bound at $`n = 0`$ holds for every $`r`$ because $`R^{\text{desc}}_{p=1}(0) = \tfrac12 \mathbb{E}\log(1 + r a_0 \|x\|^2)`$ is nonnegative; and $`R^{\text{ex}}(n)`$ is non-increasing because a predictor given $`n + 1`$ demonstrations can ignore one, and Bayes-optimal regret cannot rise with more information. Finally $`R^{\text{desc}}_{p=1}(0)`$ decreases to $`0`$ as $`r \to 0`$ by monotone convergence, which gives the two limits of the standalone regret, $`(1-p)R^{\text{ex}}(0)`$ and $`(1-p)R^{\text{ex}}(0) + H(p)`$.

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-akyurek2023" class="csl-entry">

Akyürek, Ekin, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. 2023. “What Learning Algorithm Is in-Context Learning? Investigations with Linear Models.” *ICLR*.

</div>

<div id="ref-brown2020" class="csl-entry">

Brown, Tom B. et al. 2020. “Language Models Are Few-Shot Learners.” *NeurIPS*.

</div>

<div id="ref-evans2006" class="csl-entry">

Evans, Michael, and Hadas Moshonov. 2006. “Checking for Prior-Data Conflict.” *Bayesian Analysis* 1 (4): 893–914.

</div>

<div id="ref-garg2022" class="csl-entry">

Garg, Shivam, Dimitris Tsipras, Percy Liang, and Gregory Valiant. 2022. “What Can Transformers Learn in-Context? A Case Study of Simple Function Classes.” *NeurIPS*.

</div>

<div id="ref-genewein2025" class="csl-entry">

Genewein, Tim, Li Kevin Wenliang, Jordi Grau-Moya, Anian Ruoss, Laurent Orseau, and Marcus Hutter. 2025. “Understanding Prompt Tuning and in-Context Learning via Meta-Learning.” *NeurIPS*.

</div>

<div id="ref-gupta2025" class="csl-entry">

Gupta, Ritwik, Rodolfo Corona, Jiaxin Ge, et al. 2025. “Enough Coin Flips Can Make LLMs Act Bayesian.” *ACL*.

</div>

<div id="ref-honda2025" class="csl-entry">

Honda, Ukyo, Soichiro Murakami, and Peinan Zhang. 2025. “Distilling Many-Shot in-Context Learning into a Cheat Sheet.” *Findings of EMNLP*.

</div>

<div id="ref-huangge2025" class="csl-entry">

Huang, Ruomin, and Rong Ge. 2025. “Task Descriptors Help Transformers Learn Linear Models in-Context.” *ICLR*.

</div>

<div id="ref-lescao2021" class="csl-entry">

Le Scao, Teven, and Alexander M. Rush. 2021. “How Many Data Points Is a Prompt Worth?” *NAACL*.

</div>

<div id="ref-lin2025" class="csl-entry">

Lin, Ziqian, Shubham Kumar Bharti, and Kangwook Lee. 2025. “In-Context Learning with Hypothesis-Class Guidance.” *arXiv:2502.19787*.

</div>

<div id="ref-lin2024" class="csl-entry">

Lin, Ziqian, and Kangwook Lee. 2024. “Dual Operating Modes of in-Context Learning.” *ICML*.

</div>

<div id="ref-morita2008" class="csl-entry">

Morita, Satoshi, Peter F. Thall, and Peter Müller. 2008. “Determining the Effective Sample Size of a Parametric Prior.” *Biometrics* 64 (2): 595–602.

</div>

<div id="ref-muirhead1982" class="csl-entry">

Muirhead, Robb J. 1982. *Aspects of Multivariate Statistical Theory*. Wiley.

</div>

<div id="ref-neuenschwander2020" class="csl-entry">

Neuenschwander, Beat, Sebastian Weber, Heinz Schmidli, and Anthony O’Hagan. 2020. “Predictively Consistent Prior Effective Sample Sizes.” *Biometrics* 76 (2): 578–87.

</div>

<div id="ref-panwar2024" class="csl-entry">

Panwar, Madhur, Kabir Ahuja, and Navin Goyal. 2024. “In-Context Learning Through the Bayesian Prism.” *ICLR*.

</div>

<div id="ref-raventos2023" class="csl-entry">

Raventós, Allan, Mansheej Paul, Feng Chen, and Surya Ganguli. 2023. “Pretraining Task Diversity and the Emergence of Non-Bayesian in-Context Learning for Regression.” *NeurIPS*.

</div>

<div id="ref-reimherr2021" class="csl-entry">

Reimherr, Matthew, Xiao-Li Meng, and Dan L. Nicolae. 2021. “Prior Sample Size Extensions for Assessing Prior Impact and Prior-Likelihood Discordance.” *Journal of the Royal Statistical Society: Series B* 83 (3): 413–37.

</div>

<div id="ref-reznik2026" class="csl-entry">

Reznik, Yuriy A. 2026. “The Optimal Discounting Parameter of the Power Prior Under Predictive Log-Loss.” *arXiv:2608.12159*.

</div>

<div id="ref-vonrosen1988" class="csl-entry">

Rosen, Dietrich von. 1988. “Moments for the Inverted Wishart Distribution.” *Scandinavian Journal of Statistics* 15 (2): 97–109.

</div>

<div id="ref-schmidli2014" class="csl-entry">

Schmidli, Heinz, Sandro Gsteiger, Satrajit Roychoudhury, Anthony O’Hagan, David Spiegelhalter, and Beat Neuenschwander. 2014. “Robust Meta-Analytic-Predictive Priors in Clinical Trials with Historical Control Information.” *Biometrics* 70 (4): 1023–32.

</div>

<div id="ref-tong2026" class="csl-entry">

Tong, X., Y. Zeng, and J. Zhang. 2026. “Demonstrations, CoT, and Prompting: A Theoretical Analysis of ICL.” *arXiv:2603.19611*.

</div>

<div id="ref-webson2022" class="csl-entry">

Webson, Albert, and Ellie Pavlick. 2022. “Do Prompt-Based Models Really Understand the Meaning of Their Prompts?” *NAACL*.

</div>

<div id="ref-xie2022" class="csl-entry">

Xie, Sang Michael, Aditi Raghunathan, Percy Liang, and Tengyu Ma. 2022. “An Explanation of in-Context Learning as Implicit Bayesian Inference.” *ICLR*.

</div>

<div id="ref-zhu2026" class="csl-entry">

Zhu, Qingyang, Eric Karl Oermann, and Kyunghyun Cho. 2026. “Multi-Task Bayesian in-Context Learning.” *ICML*.

</div>

</div>
