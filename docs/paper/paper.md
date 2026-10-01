# How Many In-Context Examples Is a Task Description Worth?

Bruce Quan Nguyen  
*\[Affiliation\]*

*Draft of 1 October 2026*

## Abstract

Theories of in-context learning (ICL) model a prompt’s examples, not its task description. We ask how many examples a description is worth. In a Bayesian reading of ICL both are data about a latent task. In in-context linear regression, we define a description’s effective sample size (ESS) as the number of examples that lowers a Bayes-optimal predictor’s one-step regret by as much as the description does, alone or with $`n`$ examples already in hand. Its precision $`r`$ is the fraction of prior variance it leaves, its reliability $`p`$ the probability that it is correct. A reliable description has ESS about $`(d-1)(1-r) + 1/(r a_0)`$ in $`d`$ dimensions, with $`a_0`$ the prior signal-to-noise ratio: a high-SNR term plus $`1/(r a_0)`$, the classical effective sample size of its prior, which is also its ESS once examples are plentiful. Reliability caps the value of precision only while nothing can check the description. Alone, an unreliable description keeps at least a fraction $`1-p`$ of the regret of predicting from the prior alone, so its ESS saturates: at $`p = 0.9`$ in sixteen dimensions, a hundredfold gain in precision adds ten examples, a hundred when the description is reliable. Examples verify the description, and its ESS rises toward a fraction $`p`$ of a reliable description’s. In one training run per setting, small meta-trained Transformers reproduce the Bayes-optimal ESS of a reliable description alone and under-value an unreliable one at both precisions, with implied trust $`0.85`$ against $`0.9`$; one trained only on reliable descriptions trusts an unreliable one fully, so a precise one is then worth less than nothing. Our claims concern Bayes-optimal predictors and small meta-trained sequence models, not large language models.

## 1 Introduction

<img src="figs/overview.svg" />

**Figure 1.** **A task description is worth a number of in-context examples, set by its precision and capped by its reliability.** (a) We read a prompt as Bayesian inference over a latent task $`w`$ under the pretraining prior: the description is a direct noisy observation of $`w`$ and the examples are indirect ones. The description’s effective sample size (ESS) is the number of examples $`k`$ that lowers a Bayes-optimal predictor’s one-step regret by as much as the description does. (b) Three descriptions and their ESS. The precision $`r`$ sets how narrowly the prior concentrates around the stated $`m`$; the reliability $`p`$ is the probability that the description is correct, and otherwise $`w`$ follows the base prior (dashed; its share $`1-p`$ shaded pink). Prior shapes are sketches; ESS values are from Table [3](#tab:reliability) in Appendix [A](#app:tables) ($`d = 16`$, $`a_0 = 10`$).

In-context learning (ICL) lets a model adapt to a new task from a prompt (Brown et al. 2020), which usually pairs a natural-language task description with a few demonstrations; formal theories of ICL model only the demonstrations (Garg et al. 2022; Akyürek et al. 2023; Xie et al. 2022). The evidence on descriptions is mixed: a good one can be worth hundreds of data points (Le Scao and Rush 2021; Honda et al. 2025), a misleading one can perform about as well as an informative one (Webson and Pavlick 2022), and enough examples override an explicitly stated bias (Gupta et al. 2025). Theoretical models with a task descriptor (Huang and Ge 2025; Lin et al. 2025; Tong et al. 2026) do not put its value on the same scale as the examples.

We quantify a task description’s value as the number of in-context examples it can replace. In the Bayesian reading of ICL (Xie et al. 2022), the pretraining distribution is the prior over a latent task and everything in the prompt is data: the examples observe the task indirectly through inputs, the description directly, if imperfectly. Conditioning on the description first gives the prior under which the examples are read, which is where the phrase “the description is the prior” comes from. Bayesian statistics measures the information in a prior by its *effective sample size* (ESS), the number of observations that would carry as much (Morita et al. 2008); the ESS depends on how concentrated the prior is and on whether it conflicts with the data (Evans and Moshonov 2006). For a task description these are its **precision**, how far it narrows the prior, and its **reliability**, the probability that it is correct.

#### Contributions.

1.  In noisy in-context linear regression (Garg et al. 2022; Akyürek et al. 2023; Raventós et al. 2023), where the Bayes-optimal predictor is exact, we define a description’s ESS by matching the predictor’s one-step log-loss regret given the description to that given $`n`$ examples, with or without further examples in hand; it is the prior sample size of Reimherr et al. (2021) with regret as the uncertainty measure (§[3](#sec:setting)).

2.  An unreliable description’s regret is bounded below by $`1-p`$ times the regret of predicting from the prior alone, and the bound is tight up to the entropy $`H(p)`$, so its ESS saturates however precise it is. Examples verify the description and lift the bound: its ESS with $`n`$ examples in hand rises with $`n`$ toward a fraction $`p`$ of a reliable description’s, whose own ESS falls (§[5](#sec:reliability)).

3.  A reliable description ($`p = 1`$) has ESS $`\approx (d-1)(1-r) + 1/(r a_0)`$, with $`d`$ the dimension, $`r`$ its precision, and $`a_0`$ the prior SNR: a high-SNR term plus the conjugate-prior ESS, within about two examples across the grid (§[4](#sec:precision)).

4.  Small Transformers meta-trained on this process, one training run per setting, reproduce the Bayes-optimal ESS of a reliable description alone and under-value an unreliable one at both precisions, with implied trust $`0.85`$ against $`0.9`$; trained only on reliable descriptions, they do not discount an unreliable one, and a precise one then hurts (§[6](#sec:networks)).

## 2 Related Work

#### Bayesian Perspectives on ICL.

ICL can be framed as implicit Bayesian inference (Xie et al. 2022). In linear regression, meta-trained Transformers effectively implement Bayes-optimal ridge regression (Akyürek et al. 2023; Raventós et al. 2023; Panwar et al. 2024). Genewein et al. (2025) score meta-trained predictors by their expected excess log loss (regret) against the oracle that knows the task and compare the curve with the Bayes-optimal predictor’s; we adopt both. Zhu et al. (2026) show that a Transformer given a prefix of related datasets performs amortised hierarchical Bayesian prediction, matching the oracle across prior families, which is the closest demonstration that a prior-carrying prefix is used Bayes-optimally; they do not express the prefix’s value in examples or model an unreliable one. None of these works translate the influence of a description into an equivalent number of in-context examples.

#### Task Descriptions in ICL Theory.

Recent studies have begun to model task descriptions theoretically. Huang and Ge (2025) include a descriptor token carrying the input mean; it helps their learner by centring the inputs but carries no information about the weights, so to a Bayes-optimal predictor its ESS is zero. Lin et al. (2025) guide models with hypothesis classes, and Tong et al. (2026) show that a prediction’s dependence on the instruction, right or wrong, decays exponentially in the number of demonstrations. Lin and Lee (2024) analyze Gaussian-mixture task priors whose components are fixed by pretraining; ours has two components centred by the description. None of these works quantify the description’s value as an effective sample size.

#### Effective Sample Size.

In statistics, Morita et al. (2008) define the ESS of a prior by matching its information to that of a baseline prior plus a sample; Reimherr et al. (2021) make the prior sample size a function of the data in hand, $`M(k)`$, allow it to be negative for a prior in conflict with the likelihood, and leave prediction-based uncertainty measures as an open direction; Neuenschwander et al. (2020) note that a mixture prior’s ESS is not the weighted average of its components’. Reznik (2026) defines a borrowed ESS under predictive log loss for a power prior, with a divergence between historical and current data in the role our $`1-p`$ plays; it has no mixture and does not depend on the current data. Our ESS is the prior sample size of Reimherr et al. (2021) with one-step predictive regret as the uncertainty measure, so it depends on the examples in hand. Our model of reliability is the robust mixture prior of Schmidli et al. (2014), used in clinical trials to incorporate possibly conflicting historical data: their worked example puts weight $`0.1`$ on a vague component, which is our $`p = 0.9`$, and our base prior plays the vague component’s role. In that literature’s terms, a wrong description is a prior-data conflict (Evans and Moshonov 2006).

## 3 Problem Formulation

#### Task Distribution.

We consider tasks defined by a weight vector $`w \in \mathbb{R}^d`$. An in-context example consists of an input $`x \sim \mathcal{N}(0, I_d)`$ and a corresponding label $`y = w^\top x + \epsilon`$, where the noise is $`\epsilon \sim \mathcal{N}(0, \sigma_y^2)`$. The base prior over tasks is $`w \sim \mathcal{N}(0, s_0^2 I_d)`$. For simplicity, we set $`\sigma_y = 1`$ and define the prior signal-to-noise ratio (SNR) as $`a_0 = s_0^2 / \sigma_y^2`$. We take $`a_0 = 10`$, above the values of $`1`$ to $`4`$ in the noisy settings of Garg et al. (2022; Raventós et al. 2023; Panwar et al. 2024) and within the range swept by Akyürek et al. (2023), so that a precise description can be worth many examples.

#### Task Description Model.

We model a task description as a vector $`m \in \mathbb{R}^d`$ that provides information about $`w`$. We measure the description’s *precision* by $`r = s_\ell^2 / s_0^2 \in (0, 1)`$, the ratio of its variance to the prior’s: the variance around the description is $`s_\ell^2 = r s_0^2`$, so a smaller $`r`$ is a more precise description. ($`r`$ is a variance ratio; $`1/r`$ is the ratio of the two precisions in the inverse-variance sense.) The description is generated hierarchically via $`m \sim \mathcal{N}(0, (1-r)s_0^2 I_d)`$, ensuring that the marginal distribution of $`w`$ matches the base prior. A description is not always correct. We define its *reliability* $`p`$ as the probability that the description is accurate. If correct, $`w \sim \mathcal{N}(m, s_\ell^2 I_d)`$; if incorrect (with probability $`1-p`$), $`w`$ simply follows the base prior $`\mathcal{N}(0, s_0^2 I_d)`$ independently of $`m`$. A wrong description is thus unrelated to the task, as in robust mixture priors, rather than deliberately misleading: a description that pointed away from $`w`$ by rule would be information about $`w`$ to a learner that knew the rule, and harm only a learner that did not. Consequently, the prior conditioned on the description is the robust mixture prior of Schmidli et al. (2014), $`p\,\mathcal{N}(m, s_\ell^2 I_d) + (1-p)\,\mathcal{N}(0, s_0^2 I_d)`$, with the base prior as its vague component and $`1-p`$ the weight on that component. We call a description with $`p = 1`$ *reliable* and one with $`p < 1`$ *unreliable*. Equivalently, $`m`$ is a noisy observation of the weights themselves, $`m \mid w \sim \mathcal{N}((1-r)\,w,\; r(1-r)\,s_0^2 I_d)`$ when correct, so the description is data about $`w`$ and the ESS is an exchange rate between direct observations of $`w`$ and indirect ones through $`x`$. A description with precision $`r`$ locates each weight to within $`\sqrt{r}`$ of its typical size $`s_0`$: $`32\%`$ at $`r = 0.1`$, $`10\%`$ at $`r = 0.01`$, $`3\%`$ at $`r = 0.001`$. The formulation treats the description as a numerical observation rather than text to be parsed.

#### Regret and Effective Sample Size.

We score predictors by one-step predictive log loss on a new query $`x`$. The regret is the expected excess log loss over the oracle that knows $`w`$ exactly, the expectation taken over tasks, inputs, and labels: the per-step version of the regret of Genewein et al. (2025). Log loss, rather than the squared error of Garg et al. (2022), is a proper scoring rule and the loss under which the mixture bound of §[5](#sec:reliability) holds. The prior sample size of Reimherr et al. (2021) measures a prior $`\pi`$ against a baseline $`\pi_b`$ given data $`I`$ by the $`M`$ that solves $`U_{\pi_b}(I + M) = U_\pi(I)`$: the baseline with $`M`$ further observations is as uncertain as the prior with the data. We take the base prior as $`\pi_b`$, the description-conditioned prior as $`\pi`$, the $`n`$ examples as $`I`$, and regret as the uncertainty $`U`$.

<div id="def:ess" class="definition">

**Definition 1** (Effective sample size of a description). *Fix $`p`$ and $`r`$. Let $`R^{\mathrm{desc}}(n)`$ be the regret of the Bayes-optimal predictor given the description and $`n`$ examples, and $`R^{\mathrm{ex}}(n)`$ its regret given $`n`$ examples and no description. The description’s ESS with $`n`$ examples in hand is $`\mathrm{ESS}_n = n^* - n`$, where $`n^*`$ (interpolated linearly between integers) solves $`R^{\mathrm{ex}}(n^*) = R^{\mathrm{desc}}(n)`$: the further examples an examples-only predictor needs to match it. This is the data-dependent prior sample size $`M(n)`$ above, and it is negative when the description hurts. The *effective sample size* of the description is $`\mathrm{ESS}= \mathrm{ESS}_0`$, its ESS alone.*

</div>

The ESS compares prompts of two kinds, the description alone against examples alone, and prices the description on its own in the currency of examples on their own. $`\mathrm{ESS}_n`$ asks the same with $`n`$ examples already present; §[5](#sec:reliability) shows why that matters for unreliable descriptions. Since we compare Bayes-optimal predictors under different conditioning sets, the ESS measures information content rather than the specific architecture of a learner.

## 4 The Value of Precision

We first analyze a reliable description ($`p=1`$). Given $`n`$ examples $`X_n \in \mathbb{R}^{n \times d}`$, the posterior covariance under a Gaussian prior with SNR $`a`$ is $`\Sigma = \sigma_y^2(I/a + X_n^\top X_n)^{-1}`$. This is independent of the examples’ labels and of the description $`m`$. The one-step regret simplifies to $`\frac{1}{2}\mathbb{E}\log(1 + x^\top \Sigma x)`$. A reliable description corresponds to $`n=0`$ and $`a = r a_0`$, whereas examples alone use $`a = a_0`$.

<div id="prop:precision" class="proposition">

**Proposition 1** (ESS of a reliable description). *Let $`p = 1`$, let $`\psi`$ be the digamma function, and let $`c = 1/(r a_0) = \sigma_y^2 / s_\ell^2`$: the ESS of the description’s prior $`\mathcal{N}(m, s_\ell^2 I_d)`$ in the conjugate sense, noise variance over prior variance (Neuenschwander et al. 2020), and the limit of $`\mathrm{ESS}_n`$ as $`n \to \infty`$ (§[5](#sec:reliability)). We call it the conjugate-prior ESS.*

1.  *Coarse descriptions.* Suppose $`c \le 1`$. Then for every $`n < d`$, $`R^{\mathrm{ex}}(n) \ge \tfrac12\big[\log(2a_0) + \psi\big(\tfrac{d-n}{2}\big)\big]`$, with equality in the limit $`a_0 \to \infty`$, and as $`rd \to \infty`$, uniformly in $`c \le 1`$,
    ``` math
    \mathrm{ESS}= (d-1)(1-r) + (1-r)\,c + O\!\left(\tfrac{1}{rd}\right).
    ```

2.  *Precise descriptions.* Suppose $`c \ge d`$. Then for every $`n \ge d`$, $`R^{\mathrm{ex}}(n) \le \Phi(n) := \tfrac12\big[\psi\big(\tfrac{n+1}{2}\big) - \psi\big(\tfrac{n-d+1}{2}\big)\big]`$, with equality in the limit $`a_0 \to \infty`$, and as $`c/d \to \infty`$, uniformly in $`a_0 \ge 1`$,
    ``` math
    \mathrm{ESS}= c + d + 1 - \tfrac{1}{a_0} + O\!\left(\tfrac{d}{c}\right).
    ```

*For every $`d`$, $`a_0`$, and $`r`$, $`\mathrm{ESS}\le c + d + 2`$.*

</div>

*Idea.* Regret is half the log of the leftover variance in the prediction, and a description with precision $`r`$ leaves $`r a_0 = 1/c`$ of the noise variance in each of $`d`$ directions. Examples remove variance in two ways. The first $`n < d`$ of them each kill one direction outright at high SNR, so the leftover is $`a_0`$ times a $`\chi^2_{d-n}`$; matching it to the description’s $`r a_0 \chi^2_d`$ gives $`d - n \approx rd`$, the high-SNR term $`(d-1)(1-r)`$. Beyond $`n = d`$ every direction is seen and the leftover is $`\chi^2_d / \chi^2_{n-d+1}`$, with mean $`d/(n-d-1)`$; matching it to $`d/c`$ gives $`n \approx c + d + 1`$. The base prior itself has conjugate-prior ESS $`1/a_0`$, which is the $`-1/a_0`$ in (ii) and, after the factor $`1-r`$, the $`c - 1/a_0`$ in (i). The proof, with the error terms, is in Appendix [B](#app:precision).

Both regimes say the same thing: the ESS of a description alone is its high-SNR term plus its conjugate-prior ESS, $`\mathrm{ESS}\approx (d-1)(1-r) + c`$, within two examples. In (i) the conjugate term enters as $`(1-r)c`$, which is less than $`c`$ by $`1/a_0 < 1`$; in (ii) the exact constant is $`c + d + 1 - 1/a_0`$, which exceeds $`(d-1)(1-r) + c`$ by $`2 - 1/a_0 + (d-1)r \in [1, 2)`$ up to the stated error. The two regimes meet at $`c = 1`$, where the description pins $`w`$ down as tightly as the label noise does, and at $`c \approx d`$ the ESS passes $`d`$: below that, each gain in precision removes directions and the ESS stays under $`d`$; above it, every direction is already seen and the ESS grows like $`c`$, the examples needed to match the description’s variance, plus the $`d + 1`$ that $`d`$ noisy examples cannot supply, minus the $`1/a_0`$ examples the base prior is already worth. The universal bound $`\mathrm{ESS}\le c + d + 2`$ holds everywhere, including the crossover $`1 < c < d`$ where neither expansion applies. On the grid of Table [2](#tab:precision) and Figure [2](#fig:ess)(a), the additive rule is within $`2.2`$ examples of the exact ESS in every cell with $`a_0 \ge 10`$; at $`d = 64`$, $`a_0 = 100`$, $`r = 0.1`$ the exact ESS is $`56.85`$ against $`56.79`$ from (i), and at $`d = 16`$, $`a_0 = 10`$, $`r = 0.001`$ it is $`116.8`$ against $`116.9`$ from (ii).

In words: a description that removes nine tenths of the prior variance has an ESS of nine tenths of the dimension plus its conjugate-prior ESS of one example, $`14.9`$ at $`d = 16`$ and $`57.7`$ at $`d = 64`$; one that leaves a thousandth has its conjugate-prior ESS of $`100`$ plus $`d + 1`$, $`117`$ at $`d = 16`$. Table [2](#tab:precision) (Appendix [A](#app:tables)) lists the exact values and Figure [2](#fig:ess)(a) traces the curve at $`d = 16`$.

## 5 The Bottleneck of Reliability

In reality, descriptions can be flawed or ambiguous. With an unreliable description the Bayes-optimal predictor’s posterior predictive is a mixture of two components, one that trusts the description and one that ignores it, weighted by the posterior probability that the description is correct. The following bounds measure what the hedge costs, with and without examples to check the description against.

<div id="prop:floor" class="proposition">

**Proposition 2**. Fix $`n \ge 0`$, $`p \in (0,1)`$, and $`r \in (0, 1)`$. Let $`R^{\mathrm{desc}}(n)`$ be the regret of the Bayes-optimal predictor given a description with reliability $`p`$ and precision $`r`$ and $`n`$ examples, $`R^{\mathrm{desc}}_{p=1}(n)`$ the same for a reliable description, and $`R^{\mathrm{ex}}(n)`$ its regret given $`n`$ examples and no description. Let $`Z`$ indicate that the description is correct, $`D_n`$ be the prompt (the description, the $`n`$ examples, and the query), $`\pi_n = P(Z = 1 \mid D_n)`$ the predictor’s posterior probability that the description is correct, and $`\pi_n(Z)`$ the posterior weight on the true component, $`\pi_n`$ if $`Z = 1`$ and $`1 - \pi_n`$ otherwise. Then
``` math
p\,R^{\mathrm{desc}}_{p=1}(n) + (1-p)\,R^{\mathrm{ex}}(n)
\;\le\; R^{\mathrm{desc}}(n) \;\le\;
p\,R^{\mathrm{desc}}_{p=1}(n) + (1-p)\,R^{\mathrm{ex}}(n) + \mathbb{E}[-\log \pi_n(Z)].
```
The last term, the identification term, is the conditional entropy $`H(Z \mid D_n)`$ of the component indicator given the prompt, the cost of not knowing which component is right: it equals $`H(p)`$, the binary entropy in nats, at $`n = 0`$ and is non-increasing in $`n`$. In particular, alone ($`n = 0`$) the regret is at least $`(1-p)\,R^{\mathrm{ex}}(0)`$ for every $`r`$, so the ESS cannot exceed the $`n`$ at which $`R^{\mathrm{ex}}(n) = (1-p)\,R^{\mathrm{ex}}(0)`$, and as $`r \to 0`$ the regret lies between $`(1-p)\,R^{\mathrm{ex}}(0)`$ and $`(1-p)\,R^{\mathrm{ex}}(0) + H(p)`$.

</div>

*Idea.* Compare the predictor with a twin that is also told whether the description is correct. The twin’s regret is $`p\,R^{\mathrm{desc}}_{p=1}(n) + (1-p)\,R^{\mathrm{ex}}(n)`$. The predictor cannot beat the twin, since extra information never hurts a Bayes-optimal predictor. And its posterior predictive puts weight $`\pi_n`$ on the twin’s answer, so it loses at most $`-\log \pi_n(Z)`$: with no examples, $`\log(1/p)`$ nats when the description is correct and $`\log(1/(1-p))`$ when it is wrong, which average to $`H(p)`$; more examples can only sharpen the posterior on $`Z`$ in expectation. The upper bound is the standard bound for a mixture forecaster under log loss, which pays at most the log of one over the weight it put on the right component (Cesa-Bianchi and Lugosi 2006, Cor. 3.1 and §9.2) (Merhav and Feder 1998, sec. V), applied to the two components, and the lower bound is the Bayes-optimality of the informed twin; what is new is what they say about the ESS. The proof is in Appendix [C](#app:floor).

#### Alone.

With no examples, nothing can identify the correct component: the predictor hedges by $`1-p`$ and pays up to $`H(p)`$ for not knowing which case it is in, however precise the description. For $`d=16`$ and $`a_0=10`$, $`R^{\mathrm{ex}}(0) = 2.51`$ nats and the reliable description’s regret at $`r = 0.001`$ is $`0.07`$; at $`p=0.9`$ the two bounds are $`0.32`$ and $`0.64`$, and the computed regret is $`0.57`$. So an unreliable description can replace only a bounded number of examples, and fewer than the lower bound alone allows: at $`p = 0.9`$ the lower bound alone would allow $`40`$ examples, but the identification term is all of $`H(p)`$ in the precise limit, because a description that pins $`w`$ exactly or is simply wrong leaves nothing to tell the two cases apart, and the ESS saturates at $`25`$ as $`r \to 0`$ (dotted in Figure [2](#fig:ess)); at $`p = 0.99`$ the lower bound allows $`325`$ and the limit is $`135`$. Figure [2](#fig:ess) and Table [3](#tab:reliability) (Appendix [A](#app:tables)) show the saturation across precision. For a reliable description, refining $`r`$ from $`0.1`$ to $`0.001`$ raises the ESS from $`14.9`$ to $`116`$; at $`p=0.9`$ it only grows from $`13.1`$ to $`23.1`$, while the reliable description’s keeps rising. Even a 1% unreliability ($`p=0.99`$) nearly halves the ESS of the most precise description, $`64`$ against $`116`$, while leaving the coarse one almost untouched: the curve stays within a tenth of the reliable one down to $`r = 0.01`$. The same saturation appears at $`d \in \{4, 64\}`$ and $`a_0=100`$.

#### With examples in hand.

Examples change the picture: they can reveal whether the description is correct, and once they have, the predictor can use its full precision. In Proposition [2](#prop:floor) the lower bound $`(1-p)\,R^{\mathrm{ex}}(n)`$ decreases with $`n`$ and the identification term cannot rise, so the cap on the description’s ESS rises as examples accumulate. How far it rises has a simple form once $`n`$ is large enough that the examples have identified the description and $`R^{\mathrm{ex}}(n) \approx d/(2n)`$. Identification requires the two components to be distinguishable: the identification term tends to the residual entropy of $`Z`$ given $`w`$ and $`m`$, which is negligible here because the components are far apart (their KL divergence is of order $`\tfrac{d}{2}\log(1/r)`$) but stays positive for a coarse description, $`0.17`$ nats at $`d = 5`$, $`r = 0.5`$. Writing $`c = 1/(r a_0)`$, the reliable-description regret is then about $`d/(2(n + c))`$, so the lower bound is $`\tfrac{d}{2}\big[\tfrac{p}{n+c} + \tfrac{1-p}{n}\big]`$, and solving $`d/(2n^*)`$ equal to it gives
``` math
\mathrm{ESS}_n \;\approx\; \frac{n\,p\,c}{\,n + (1-p)\,c\,} \;\xrightarrow[n \to \infty]{}\; p\,c = \frac{p}{r a_0},
```
against $`\mathrm{ESS}_n \to c = 1/(r a_0)`$ for a reliable description: the conjugate-prior ESS is the many-examples limit. So a precise unreliable description, once examples have verified it, has $`p`$ times the ESS of a reliable one; alone, the $`n = 0`$ case caps it far lower. The formula applies only once $`n`$ exceeds $`d`$, and the Bayes-optimal predictor shows two regimes before that limit (Figure [2](#fig:ess)(c); $`r = 0.001`$, $`p = 0.9`$, $`d = 16`$, $`c = 100`$). The first one or two examples identify the description: the identification term falls from $`H(p) = 0.33`$ nats to $`0.07`$ after one example and $`0.02`$ after two, and the ESS rises from $`23`$ examples alone to $`30`$ after one. It then stays near $`30`$ until the examples outnumber the dimension, because the wrong-description component still carries the full examples-only regret, and rises once $`n > d`$ as that component’s regret shrinks: $`29`$ at $`n = 10`$, $`37`$ at $`16`$, $`88`$ at $`100`$, against the formula’s $`82`$ at $`n = 100`$ and a limit of $`90`$. The reliable description’s ESS drifts from $`117`$ to $`108`$ over the same range, and at $`r = 0.01`$, where $`c = 10`$ is below $`d`$, the unreliable description gains one example after the first example and then loses ESS like the reliable one. Figure [2](#fig:ess) shows this across precision: with $`100`$ examples in hand the most precise description has ESS $`88`$ at $`p = 0.9`$, four fifths of the reliable description’s, where alone it had $`23`$ against $`117`$. Reliability caps precision until examples can check the description, and then costs a factor $`p`$.

<img src="../../results/rq2-reliability/ess.svg" />

**Figure 2.** **Reliability caps precision only while nothing can check the description.** ESS of a description against its precision $`r`$ with $`n`$ examples in hand ($`\mathrm{ESS}_n`$ of Definition [1](#def:ess)), one panel per reliability; $`d = 16`$, $`a_0 = 10`$, log scales, one seed of 8,000 prompts. Dotted: the limit of the $`n = 0`$ ESS as $`r \to 0`$ at $`p = 0.9`$, $`25`$ examples. Alone, the $`p = 0.9`$ description saturates at $`23`$ examples while the reliable one keeps rising; with $`100`$ examples in hand the three panels nearly coincide ($`109`$, $`107`$, $`88`$ at $`r = 0.001`$).

#### A fully trusting learner.

The cap and its lifting assume a predictor calibrated to $`p`$. A predictor that takes the description as certainly correct, trust $`q = 1`$ when the truth is $`p = 0.9`$, pays for the wrong tenth in full: its posterior predictive is as tight as the description, so when the description is wrong the answer lies far outside it. At $`d = 16`$, $`a_0 = 10`$, the description alone costs this predictor $`2.2`$ nats at $`r = 0.1`$, where no description costs $`2.51`$, so a coarse description still helps ($`6.5`$ examples against $`13.2`$ when calibrated). At $`r = 0.01`$ it costs $`6.3`$ nats and at $`r = 0.001`$ $`13.4`$: worse than no description at all. Examples repair this only slowly, because the wrong tenth carries a prior as tight as the description’s: after $`100`$ examples the $`r = 0.01`$ description still leaves the predictor $`58`$ examples behind one that had no description, and the $`r = 0.001`$ description leaves $`4.0`$ nats of regret against $`0.09`$ without it. To a learner that cannot doubt the description, precision is a liability. §[6](#sec:networks) shows this learner trained: a network that has never seen a wrong description behaves like it.

## 6 Meta-Trained Transformers

#### Design.

We meta-train Transformers on prompts from the process of §[3](#sec:setting) and compare them with the Bayes-optimal predictor on the same prompts, following Genewein et al. (2025). Prompts use the prefix embedding of Huang and Ge (2025): an optional descriptor token carrying the description $`m`$, up to 20 example tokens each holding an input beside its answer, and a final query token, with two indicator coordinates marking the token type; there is no positional encoding, so the examples are exchangeable. Architecture and optimisation follow Garg et al. (2022): 12 layers, 8 heads, width 256, Adam at $`10^{-4}`$, fresh prompts at every step. The network outputs a mean and a log variance for the query’s answer and is trained on the Gaussian log loss, so its regret is on the scale of §[3](#sec:setting). We set $`d=5`$ and $`a_0=10`$ (the network sees $`w \sim \mathcal{N}(0, I)`$ and noise variance $`0.1`$, the same task in the units of Garg et al. (2022)), draw the number of examples uniformly from $`0`$ to $`20`$, and omit the description in half the prompts. The descriptor token supplies only $`m`$; each network is trained at one $`r`$ and one $`p`$, which it must learn from the training distribution. Each model trains for 40,000 steps at batch 1024 without a curriculum (Garg et al. (2022) use batch 64, 500,000 steps, and a curriculum over dimension and prompt length), one training run per setting; in a pilot the mean gap to the Bayes-optimal predictor fell from $`0.07`$ nats at 5,000 steps to $`0.03`$ at 20,000 and about $`0.02`$ from 30,000 to 40,000.

#### Readouts.

We evaluate each network on 20,000 fresh prompts for every $`n`$ from $`0`$ to $`20`$, with and without a description. As in Definition [1](#def:ess), the ESS is where the network’s regret with the description alone meets its own regret curve for examples alone. The standard error of each regret is below $`0.02`$ nats, except for the precise $`p = 1`$ model tested at $`p = 0.9`$, where it reaches $`0.10`$ with a description at small $`n`$; the examples-only curve falls by about $`0.13`$ nats per example near $`n = 0`$, so these move an ESS by at most $`0.15`$ examples, $`0.8`$ in that last case. We also evaluate the two models trained at $`p=1`$ on prompts whose descriptions are correct only with probability $`0.9`$.

<div id="tab:networks">

| $`r`$ | $`p_{\mathrm{train}}`$ | $`p_{\mathrm{test}}`$ | Network | Trained | Calibrated | Description | None |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 1 | 1 | 14.74 | 15.35 | — | 0.026 | 0.030 |
| 0.1 | 1 | 1 | 5.24 | 5.28 | — | 0.017 | 0.016 |
| 0.01 | 0.9 | 0.9 | 4.30 | 6.82 | — | 0.073 | 0.022 |
| 0.1 | 0.9 | 0.9 | 3.57 | 4.22 | — | 0.034 | 0.018 |
| 0.01 | 1 | 0.9 | 0.00 | $`\le 0`$ | 6.92 | -0.186 | 0.030 |
| 0.1 | 1 | 0.9 | 2.32 | 2.05 | 4.26 | 0.010 | 0.016 |

**Table 1.** Networks against the Bayes-optimal predictor ($`d=5`$, $`a_0=10`$, one training run per row, 20,000 evaluation prompts). “Trained” is the Bayes-optimal predictor with trust equal to the training reliability $`p_{\mathrm{train}}`$, the predictor each network is trained toward; “calibrated” is the predictor with trust equal to the test reliability, shown where the two differ. The last two columns are the mean regret gap, network minus the “trained” predictor, over $`n = 0, \dots, 20`$, with and without a description.

</div>

<img src="../../results/rq3-meta-trained/regret.svg" />

**Figure 3.** **Networks trained on the matching distribution track the Bayes-optimal predictor, except on an unreliable description alone.** Regret against the number of examples for the network (markers; standard errors over 20,000 prompts are smaller than the markers) and the Bayes-optimal predictor (lines); one training run per panel. The lower curve in each panel is prompts with a description followed by $`n`$ examples, the upper curve prompts with $`n`$ examples alone. The ESS is the $`n`$ at which the upper curve reaches the lower curve’s value at $`n = 0`$. (a) $`r=0.01`$, $`p=1`$; (b) $`r=0.1`$, $`p=1`$; (c) $`r=0.01`$, $`p=0.9`$; (d) $`r=0.1`$, $`p=0.9`$. In (c) and (d) the network’s regret with the description alone exceeds the Bayes-optimal predictor’s by $`0.5`$ and $`0.2`$ nats, and its ESS is $`4.3`$ against $`6.8`$ and $`3.6`$ against $`4.2`$.

#### Results.

For reliable descriptions the networks reproduce the Bayes-optimal values (Table [1](#tab:networks), Figure [3](#fig:networks)): ESS $`14.7`$ against $`15.4`$ for the precise description and $`5.24`$ against $`5.28`$ for the coarse one, with regret $`0.02`$ to $`0.03`$ nats above the Bayes-optimal predictor’s on average over $`n`$. Unreliable descriptions are under-valued alone at both precisions: ESS $`4.3`$ against $`6.8`$ at $`r = 0.01`$ and $`3.6`$ against $`4.2`$ at $`r = 0.1`$, with the regret from the description alone $`0.5`$ and $`0.2`$ nats above the Bayes-optimal predictor’s, and an implied trust of $`0.85`$ rather than $`0.9`$ in both, where the implied trust at $`n`$ examples is the $`q`$ whose Bayes-optimal predictor has the posterior-mean prediction closest, in mean squared difference over prompts, to the network’s. The gap closes as examples arrive, to $`0.05`$ and $`0.03`$ nats by $`n = 5`$ (Figure [3](#fig:networks)c, d), and the networks reproduce the two behaviours of §[5](#sec:reliability) with examples in hand: at $`r = 0.01`$ the network’s $`\mathrm{ESS}_n`$ rises from $`4.3`$ alone to $`7.3`$ after five examples and $`8.1`$ after ten (Bayes-optimal $`6.8`$, $`8.7`$, $`9.0`$), while at $`r = 0.1`$ it falls from $`3.6`$ to $`2.3`$ and $`1.3`$ (Bayes-optimal $`4.2`$, $`2.3`$, $`1.1`$). It is not under-training: a model trained at $`r = 0.05`$, $`p = 0.9`$ with the same recipe and continued to 80,000 steps on fresh prompts kept its ESS ($`3.9`$ against $`5.0`$) and its implied trust of $`0.85`$. With one training run per setting, we report the under-trust as an observation rather than a finding.

#### Transfer to unreliable descriptions.

The models trained at $`p=1`$ behave like a Bayes-optimal predictor that trusts the description fully. With one description in ten wrong, the model trained on precise reliable descriptions is hurt by a description: its regret with the description alone is $`2.87`$ nats against $`1.87`$ without one, so its ESS is zero where a calibrated predictor would get $`6.9`$, and its implied trust is $`0.999`$. The fully trusting Bayes-optimal predictor does worse still, $`3.32`$ nats ($`\le 0`$ in Table [1](#tab:networks)). The coarse model is forgiven most of its trust, ESS $`2.3`$ against the calibrated $`4.3`$ with implied trust $`0.95`$, because a wrong coarse description does little harm. This is the fully trusting learner of §[5](#sec:reliability) seen trained: the cost of unreliability falls on precise descriptions, and a learner that has never seen a wrong one pays it in full.

## 7 Discussion

For a Bayes-optimal predictor, a task description has a definite effective sample size in in-context examples. When the description is reliable, its ESS is the share of the dimension it pins down plus the conjugate-prior ESS $`1/(r a_0)`$, the examples that would match its variance. Otherwise the predictor must hedge against its being wrong, so a very precise description’s ESS saturates. The hedge is forced only while nothing can check the description; a few examples lift it, and a precise unreliable description then has a larger ESS than alone. Reliability is not a fixed discount but a cost that examples pay down, to a factor $`p`$. This holds for a predictor calibrated to the description’s reliability; to one that cannot doubt it, a precise description that is sometimes wrong is worse than none, and examples repair the damage slowly.

#### Limitations and Future Work.

Our claims cover Bayes-optimal predictors and small meta-trained Transformers in a linear-Gaussian setting, chosen for exact calculation; whether they carry to large language models reading text is open. The ESS scores one-step prediction: Definition [1](#def:ess) counts the examples already in the prompt, not a sequential session in which the predictor’s own feedback accumulates. Over such horizons a reliable description’s long-horizon ESS tends to the number of examples whose expected information gain matches its $`\tfrac{d}{2}\log(1/r)`$ nats, below the one-step value ($`7.8`$ against $`14.9`$ at $`d = 16`$, $`a_0 = 10`$, $`r = 0.1`$); the horizon dependence of unreliable descriptions is left for future work.

## A Exact values

Tables [2](#tab:precision) and [3](#tab:reliability) give the exact one-step ESS behind §[4](#sec:precision) and §[5](#sec:reliability) at the ends of the grids; Figure [2](#fig:ess) shows the relationships between them. The two tables are separate simulations, so the one quantity they share, the reliable description at $`d = 16`$, $`r = 0.001`$, differs in its last digit: $`117`$ against $`116`$.

<div id="tab:precision">

| $`d`$ | $`r`$ | $`\mathrm{ESS}`$ | $`(d-1)(1-r) + 1/(r a_0)`$ |
|------:|------:|-----------------:|---------------------------:|
|     4 |   0.1 |             4.41 |                        3.7 |
|     4 |  0.01 |             14.5 |                         13 |
|     4 | 0.001 |              105 |                        103 |
|    16 |   0.1 |             14.9 |                       14.5 |
|    16 |  0.01 |             26.1 |                       24.9 |
|    16 | 0.001 |              117 |                        115 |
|    64 |   0.1 |             57.7 |                       57.7 |
|    64 |  0.01 |             73.3 |                       72.4 |
|    64 | 0.001 |              164 |                        163 |

**Table 2.** Exact ESS of a reliable description ($`p=1`$) for $`a_0=10`$, three seeds (standard deviation across seeds below 2% of each value), against the additive rule $`(d-1)(1-r) + 1/(r a_0)`$ of Proposition [1](#prop:precision), which is within $`2.2`$ examples of every cell.

</div>

<div id="tab:reliability">

| $`r`$ | $`p = 1`$ | $`p = 0.99`$ | $`p = 0.9`$ |
|------:|----------:|-------------:|------------:|
|   0.1 |      14.9 |         14.6 |        13.1 |
|  0.01 |      26.1 |         24.2 |        18.4 |
| 0.001 |       116 |         64.2 |        23.1 |

**Table 3.** ESS of unreliable descriptions ($`d=16`$, $`a_0=10`$), three seeds; standard deviation across seeds at most 2% of each value. Down a column precision rises tenfold per row; across a row the description is wrong never, one time in a hundred, and one time in ten.

</div>

## B Proof of Proposition [1](#prop:precision)

<div class="proof">

*Proof.* Throughout, $`\varepsilon = 1/a_0`$, $`x`$ is the fresh query with $`A = \|x\|^2 \sim \chi^2_d`$, and $`\Sigma = (\varepsilon I + X_n^\top X_n)^{-1}`$. We use $`\mathbb{E}\log \chi^2_k = \psi(k/2) + \log 2`$, $`\mathbb{E}[1/\chi^2_k] = 1/(k-2)`$, $`\mathbb{E}[1/\chi^4_k] = 1/((k-2)(k-4))`$, and the Wishart facts that for $`W \sim W_q(p, I)`$ with $`p \ge q`$ and $`z \sim \mathcal{N}(0, I_q)`$ independent of $`W`$, $`z^\top W^{-1} z`$ is distributed as $`\chi^2_q / \chi^2_{p-q+1}`$ with independent numerator and denominator (Muirhead 1982, Theorem 3.2.12), $`\mathbb{E}W^{-1} = I/(p-q-1)`$, and $`\mathbb{E}\operatorname{tr} W^{-2} = q(p-1)/((p-q)(p-q-1)(p-q-3))`$ (Rosen 1988).

*Step 1: regret is leftover variance.* The posterior predictive for $`y`$ is the true conditional law of $`y`$ given the data, a Gaussian with variance $`1 + x^\top \Sigma x`$. The expected log loss of a true conditional law is its entropy, $`\tfrac12 \log(2\pi e (1 + x^\top \Sigma x))`$; the oracle’s predictive is $`\mathcal{N}(w^\top x, 1)`$ with entropy $`\tfrac12 \log(2\pi e)`$. So $`R^{\mathrm{ex}}(n) = \tfrac12 \mathbb{E}\log(1 + x^\top \Sigma x)`$, and since $`\Sigma`$ shrinks pathwise as $`n`$ grows, $`R^{\mathrm{ex}}`$ is nonincreasing in $`n`$. For the description, $`n = 0`$ and $`\Sigma = r a_0 I`$, so with $`g(t) := \mathbb{E}\log(1 + At)`$,
``` math
2R^{\mathrm{desc}} = g(1/c) = \log(2 r a_0) + \psi(d/2) + \delta^{\mathrm{desc}},
\qquad
\delta^{\mathrm{desc}} = \mathbb{E}\log(1 + c/A) = \frac{c}{d-2} + O\!\left(\frac{c^2}{d^2}\right),
```
the last from $`y - y^2/2 \le \log(1+y) \le y`$ with $`y = c/A`$ ($`d \ge 5`$).

*Step 2: fewer than $`d`$ examples.* Write $`X_n^\top X_n = \sum_{i \le n} \lambda_i u_i u_i^\top`$ with $`\lambda_i > 0`$ almost surely, let $`P_\perp`$ project onto the $`k = d - n`$ directions the examples did not look at, and $`Q = \|P_\perp x\|^2 \sim \chi^2_k`$. Then $`\Sigma`$ has eigenvalue $`a_0`$ on those directions and $`1/(\varepsilon + \lambda_i)`$ on the others, so
``` math
1 + x^\top \Sigma x = a_0 Q + T, \qquad T = 1 + V, \quad V = \sum_{i \le n} \frac{z_i^2}{\varepsilon + \lambda_i}, \quad z_i = u_i^\top x .
```
Given $`X_n`$, the $`z_i`$ are i.i.d. standard normal and independent of $`Q`$, and the $`\lambda_i`$ are the eigenvalues of $`X_n X_n^\top \sim W_n(d, I)`$. Hence
``` math
2R^{\mathrm{ex}}(n) = \log(2a_0) + \psi(k/2) + \delta^{\mathrm{ex}}, \qquad \delta^{\mathrm{ex}} = \mathbb{E}\log\!\left(1 + \frac{\varepsilon T}{Q}\right) \ge 0,
```
which is the lower bound in (i); $`\varepsilon T \to 0`$ as $`a_0 \to \infty`$, so $`\delta^{\mathrm{ex}} \to 0`$ by monotone convergence, which is the equality.

For the finite-$`a_0`$ term, let $`F = \sum_i z_i^2 / \lambda_i = z^\top (X_n X_n^\top)^{-1} z \sim \chi^2_n / \chi^2_{k+1}`$, so $`\mathbb{E}F = n/(k-1)`$ and $`\mathbb{E}F^2 = n(n+2)/((k-1)(k-3))`$. Then $`F - \varepsilon \sum_i z_i^2/\lambda_i^2 \le V \le F`$, and $`\mathbb{E}\sum_i z_i^2/\lambda_i^2 = \mathbb{E}\operatorname{tr}(X_n X_n^\top)^{-2} = n(d-1)/(k(k-1)(k-3))`$. Applying $`y - y^2/2 \le \log(1+y) \le y`$ with $`y = \varepsilon T / Q`$ and the independence of $`Q`$ from $`T`$,
``` math
\frac{\varepsilon\,(1 + \mathbb{E}V)}{k-2} - \frac{\varepsilon^2 \mathbb{E}T^2}{2(k-2)(k-4)} \le \delta^{\mathrm{ex}} \le \frac{\varepsilon\,(1 + \mathbb{E}F)}{k-2},
\qquad\text{so}\qquad
\delta^{\mathrm{ex}} = \frac{\varepsilon (d-1)}{(k-1)(k-2)} + O\!\left(\frac{\varepsilon^2 d^2}{k^4}\right), \qquad k \ge 5 .
```

*Step 3: solve (i).* Matching $`R^{\mathrm{ex}}(n) = R^{\mathrm{desc}}`$ cancels $`\log a_0`$:
``` math
\psi(k/2) - \psi(d/2) = \log r + \delta^{\mathrm{desc}} - \delta^{\mathrm{ex}} .
```
Read it as in the high-SNR case: the $`k`$ untouched directions must be as big, on a log scale, as $`d`$ directions each shrunk by $`r`$, with the finite-SNR leftovers $`\delta^{\mathrm{desc}}`$ and $`\delta^{\mathrm{ex}}`$ as corrections. Treat $`k`$ as real; the paper’s ESS interpolates $`R^{\mathrm{ex}}`$ linearly between integers, and since the right side of the display is, as a function of $`n`$, smooth with second derivative $`O(1/k^2)`$ and slope of order $`1/k`$, that moves the root by $`O(1/k)`$, which the error term absorbs. With $`\psi(z) = \log z - 1/(2z) + O(z^{-2})`$,
``` math
k = rd \, \exp\!\big(\eta\big), \qquad \eta = \frac1k - \frac1d + \delta^{\mathrm{desc}} - \delta^{\mathrm{ex}} + O(k^{-2}) .
```
As $`rd \to \infty`$ every term of $`\eta`$ vanishes, so $`k = rd(1 + o(1))`$ as in the high-SNR proof, and then $`1/k = O(1/(rd))`$, $`\delta^{\mathrm{desc}} = O(c/d)`$, and $`\delta^{\mathrm{ex}} = O(\varepsilon d / k^2) = O(c/(rd))`$ because $`\varepsilon = rc`$. So $`\eta = O((1+c)/(rd))`$ and $`k = rd + rd\,\eta + O((1+c)^2/(rd))`$. In $`rd\,\eta`$: $`rd/k = 1 + O((1+c)/(rd))`$; $`rd\,\delta^{\mathrm{desc}} = rc + O(rc(1+c)/d) = 1/a_0 + O((1+c)^2/(rd))`$; and $`rd\,\delta^{\mathrm{ex}} = (\varepsilon/r)(rd/k)^2(1 + O(1/k)) + O(\varepsilon^2 r d^3/k^4) = c + O(c(1+c)/(rd)) + O(c^2/(rd))`$. Hence
``` math
k = rd + 1 - r + \frac{1}{a_0} - c + O\!\left(\frac{(1+c)^2}{rd}\right),
\qquad
\mathrm{ESS}= d - k = (d-1)(1-r) + c - \frac{1}{a_0} + O\!\left(\frac{(1+c)^2}{rd}\right),
```
and $`c - 1/a_0 = c - rc = (1-r)c`$. For $`c \le 1`$ the error is $`O(1/(rd))`$, which is (i). As $`a_0 \to \infty`$ with $`r`$ fixed, $`c \to 0`$ and this is the high-SNR rule $`(d-1)(1-r) + O(1/(rd))`$.

*Step 4: at least $`d`$ examples, an exact representation.* Rotate coordinates so that $`x = \|x\| e_1`$; the rows of the rotated $`X_n`$ are still i.i.d. $`\mathcal{N}(0, I_d)`$ and independent of $`\|x\|`$. Then $`x^\top \Sigma x = A\,\Sigma_{11} = A/S`$, where by the Schur complement and the Woodbury identity, with $`X_1`$ the first column of $`X_n`$ and $`X_{-1}`$ the other $`d-1`$ columns,
``` math
S = \varepsilon + X_1^\top \big(I_n + a_0 X_{-1} X_{-1}^\top\big)^{-1} X_1 .
```
Given $`X_{-1}`$, the matrix in the middle has eigenvalue $`1`$ on the $`m = n - d + 1`$ directions orthogonal to the columns of $`X_{-1}`$ and $`1/(1 + a_0 \mu_j)`$ on the other $`d - 1`$, where $`\mu_j`$ are the eigenvalues of $`M = X_{-1}^\top X_{-1} \sim W_{d-1}(n, I)`$. Since $`X_1 \sim \mathcal{N}(0, I_n)`$ is independent of $`X_{-1}`$, its coordinates in that eigenbasis are i.i.d. standard normal, and
``` math
S = \varepsilon + Q + \sum_{j < d} \frac{z_j^2}{1 + a_0 \mu_j}, \qquad Q \sim \chi^2_m,
\quad\text{so}\quad
Q \le S \le Q + \varepsilon (1 + F), \quad F = z^\top M^{-1} z \sim \frac{\chi^2_{d-1}}{\chi^2_{n-d+2}},
```
with $`A`$, $`Q`$, and $`(z, M)`$ independent, $`\mathbb{E}F = (d-1)/(n-d)`$ and $`\mathbb{E}F^2 = (d-1)(d+1)/((n-d)(n-d-2))`$.

At $`a_0 = \infty`$, $`S = Q`$ and $`1 + A/Q = (A + Q)/Q`$ with $`A + Q \sim \chi^2_{n+1}`$, so $`\mathbb{E}\log(1 + A/Q) = \psi\big(\tfrac{n+1}{2}\big) - \psi\big(\tfrac{m}{2}\big) = 2\Phi(n)`$. Since $`S \ge Q`$ for every $`a_0`$, $`R^{\mathrm{ex}}(n) \le \Phi(n)`$, with equality as $`a_0 \to \infty`$ by monotone convergence: the first claim of (ii).

*Step 5: the universal bound.* Conditionally on $`A`$, $`\log(1 + A/Q)`$ is concave in $`1/Q`$, so Jensen gives $`2\Phi(n) \le \mathbb{E}\log\big(1 + A/(m-2)\big) = g(1/(m-2))`$ for $`m > 2`$. If $`m - 2 \ge c`$, that is $`n \ge c + d + 1`$, then $`g(1/(m-2)) \le g(1/c) = 2R^{\mathrm{desc}}`$, so $`R^{\mathrm{ex}}(n) \le \Phi(n) \le R^{\mathrm{desc}}`$. The smallest such integer is at most $`c + d + 2`$, and $`R^{\mathrm{ex}}`$ is nonincreasing, so $`\mathrm{ESS}\le c + d + 2`$ in every setting.

*Step 6: solve (ii).* Two sandwiches. First, finite $`a_0`$: with $`D = \log(1 + A/Q) - \log(1 + A/S) = \log\big(1 + \tfrac{A(S-Q)}{Q(S+A)}\big) \ge 0`$, the bounds $`0 \le S - Q \le \varepsilon(1+F)`$, $`1/(S+A) \ge 1/Q - (S + A - Q)/Q^2`$, and $`y - y^2/2 \le \log(1+y) \le y`$ give, for $`a_0 \ge 1`$ and $`n \ge d + 6`$,
``` math
\mathbb{E}D = \frac{\varepsilon d}{m^2}\Big(1 + O\big(\tfrac{d+1}{m}\big)\Big),
\qquad\text{so}\qquad
2R^{\mathrm{ex}}(n) = 2\Phi(n) - \frac{\varepsilon d}{m^2} + O\!\left(\frac{\varepsilon d^2}{m^3}\right).
```
(The upper bound is $`\mathbb{E}[\varepsilon A (1+F)/Q^2] = \varepsilon d (n-1)/((n-d)(n-d-1)(n-d-3))`$; the lower bound subtracts $`\mathbb{E}[\varepsilon A (\varepsilon(1+F) + A)/Q^3]`$ and $`\tfrac12 \mathbb{E}[\varepsilon^2 A^2 (1+F)^2 / Q^4]`$, both $`O(\varepsilon d^2 / m^3)`$.) Second, $`\Phi`$ against its mean: $`g''(t) = -\mathbb{E}[A^2/(1+At)^2] \ge -(d^2 + 2d)`$ and $`\operatorname{Var}(1/Q) = 2/((m-2)^2(m-4))`$, so a second-order Taylor bound around $`\mathbb{E}[1/Q] = 1/(m-2)`$ gives
``` math
2\Phi(n) = g\!\left(\tfrac{1}{m-2}\right) - \vartheta\,\frac{d^2 + 2d}{(m-2)^2 (m-4)}, \qquad \vartheta \in [0, 1] .
```
Together,
``` math
2R^{\mathrm{ex}}(n) - 2R^{\mathrm{desc}} = g\!\left(\tfrac{1}{m-2}\right) - g\!\left(\tfrac1c\right) - \frac{\varepsilon d}{m^2} + O\!\left(\frac{d^2}{m^3}\right).
```
Now $`d(1 - (d+2)t) \le g'(t) \le d`$ because $`A - A^2 t \le A/(1+At) \le A`$. Put $`m - 2 = c + s`$ with $`|s| \le c/2`$. Then $`g(1/(m-2)) - g(1/c) = -\frac{ds}{c(c+s)}\big(1 + O(d/c)\big)`$ and $`\varepsilon d/m^2 = \frac{\varepsilon d}{c^2}\big(1 + O((|s|+1)/c)\big)`$, so
``` math
2R^{\mathrm{ex}}(n) - 2R^{\mathrm{desc}} = -\frac{d}{c^2}\Big[s + \varepsilon + O\!\Big(\frac{(1 + |s|)(d + |s|)}{c}\Big)\Big].
```
At $`s = -\varepsilon \pm K d/c`$ the bracket has the sign $`\pm`$ once $`K`$ exceeds the constant in the $`O`$ (for $`c/d`$ large, $`|s| = O(1)`$). So $`R^{\mathrm{ex}} - R^{\mathrm{desc}}`$ changes sign between $`n = c + d + 1 - \varepsilon - Kd/c`$ and $`n = c + d + 1 - \varepsilon + Kd/c`$; linear interpolation between integers moves the crossing by $`O(1/c)`$, and the moment conditions $`n \ge d + 6`$ hold since $`n \ge c`$. Hence $`\mathrm{ESS}= c + d + 1 - 1/a_0 + O(d/c)`$, which is (ii). ◻

</div>

## C Proof of Proposition [2](#prop:floor)

<div class="proof">

*Proof.* Let $`D_n = (x_{1:n}, y_{1:n}, m, x)`$ be everything the predictor sees before predicting $`y`$, and $`Z = 1`$ if the description is correct, $`Z = 0`$ otherwise.

*Step 1: the posterior predictive is a $`\pi_n`$-mixture.* The Bayes-optimal predictor’s posterior predictive is $`q(y) = \pi_n f_1(y) + (1 - \pi_n) f_0(y)`$, where $`\pi_n = P(Z = 1 \mid D_n)`$ and $`f_1`$, $`f_0`$ are the predictives of the two components given $`D_n`$: $`f_1`$ is the reliable-description predictor’s, and $`f_0`$ is the examples-only predictor’s, because under $`Z = 0`$ the description is independent of $`w`$ and carries no information about $`y`$. At $`n = 0`$, $`\pi_0 = p`$: seeing $`m`$ says nothing about $`Z`$, because $`m`$ has the same law $`\mathcal{N}(0, (1-r) a_0 I)`$ either way, and $`x`$ is independent of $`Z`$, so there is nothing to update on. The components are then $`f_1 = \mathcal{N}(m^\top x,\, 1 + r a_0\|x\|^2)`$ and $`f_0 = \mathcal{N}(0,\, 1 + a_0\|x\|^2)`$.

*Step 2: the twin.* Given $`(D_n, Z)`$, the true conditional density of $`y`$ is $`f_Z`$. So a twin who is told $`Z`$ predicts with $`f_1`$ when $`Z = 1`$, with regret $`R^{\mathrm{desc}}_{p=1}(n)`$, and with $`f_0`$ when $`Z = 0`$, with regret $`R^{\mathrm{ex}}(n)`$. Its regret is $`p\,R^{\mathrm{desc}}_{p=1}(n) + (1-p)\,R^{\mathrm{ex}}(n)`$.

*Step 3: the predictor cannot beat the twin (lower bound).* Given $`(D_n, Z)`$, the expected log loss of any density $`q`$ exceeds that of the true density $`f_Z`$ by $`\mathbb{E}[-\log q \mid D_n, Z] - \mathbb{E}[-\log f_Z \mid D_n, Z] = \mathrm{KL}(f_Z \,\|\, q) \ge 0`$. Averaging over $`Z`$ and $`D_n`$ gives $`R^{\mathrm{desc}}(n) \ge p\,R^{\mathrm{desc}}_{p=1}(n) + (1-p)\,R^{\mathrm{ex}}(n)`$.

*Step 4: the predictor loses at most the identification term (upper bound).* Pointwise, $`q \ge \pi_n f_1`$ and $`q \ge (1 - \pi_n) f_0`$, so $`-\log q \le -\log f_Z - \log \pi_n(Z)`$ with $`\pi_n(1) = \pi_n`$ and $`\pi_n(0) = 1 - \pi_n`$. Taking expectations gives the upper bound with the identification term $`\mathbb{E}[-\log \pi_n(Z)]`$. At $`n = 0`$ it is $`p \log(1/p) + (1-p)\log(1/(1-p)) = H(p)`$.

*Step 5: the identification term is non-increasing.* $`\mathbb{E}[-\log \pi_n(Z)] = H(Z \mid D_n)`$, the conditional entropy of $`Z`$ given the data. The query $`x`$ is independent of $`Z`$ and of everything else in $`D_n`$, so $`H(Z \mid D_n) = H(Z \mid x_{1:n}, y_{1:n}, m)`$, and these conditioning sets are nested in $`n`$. Conditioning on more does not increase conditional entropy, so the term is non-increasing.

*Step 6: the $`n = 0`$ case.* $`R^{\mathrm{desc}}_{p=1}(0) = \tfrac12 \mathbb{E}\log(1 + r a_0 \|x\|^2)`$ is nonnegative, which gives the lower bound $`(1-p)\,R^{\mathrm{ex}}(0)`$ for every $`r`$. The ESS is the $`n`$ at which $`R^{\mathrm{ex}}(n) = R^{\mathrm{desc}}(0) \ge (1-p)\,R^{\mathrm{ex}}(0)`$, and $`R^{\mathrm{ex}}(n)`$ is non-increasing in $`n`$ (a predictor given $`n + 1`$ examples can ignore one, and Bayes-optimal regret cannot rise with more information), so the ESS cannot exceed the $`n`$ at which $`R^{\mathrm{ex}}(n) = (1-p)\,R^{\mathrm{ex}}(0)`$. Finally $`R^{\mathrm{desc}}_{p=1}(0)`$ decreases to $`0`$ as $`r \to 0`$ by monotone convergence, which gives the two limits. ◻

</div>

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-akyurek2023" class="csl-entry">

Akyürek, Ekin, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. 2023. “What Learning Algorithm Is in-Context Learning? Investigations with Linear Models.” *ICLR*.

</div>

<div id="ref-brown2020" class="csl-entry">

Brown, Tom B. et al. 2020. “Language Models Are Few-Shot Learners.” *NeurIPS*.

</div>

<div id="ref-cesabianchi2006" class="csl-entry">

Cesa-Bianchi, Nicolò, and Gábor Lugosi. 2006. *Prediction, Learning, and Games*. Cambridge University Press.

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

<div id="ref-merhav1998" class="csl-entry">

Merhav, Neri, and Meir Feder. 1998. “Universal Prediction.” *IEEE Transactions on Information Theory* 44 (6): 2124–47.

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
