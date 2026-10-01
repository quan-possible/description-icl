# Experiments

This file is the single description of how the experiments are set up. When
the code and this file disagree, one of them is wrong and gets fixed. A design
choice changes only by editing the [decision table](#decisions) here first,
with the date and the reason.

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, Clear Sans, Noto Sans, Helvetica Neue, Arial, Noto Sans CJK JP, sans-serif","fontSize":"16px","primaryTextColor":"#171717","lineColor":"#a8a8a8","mainBkg":"#ffffff","clusterBkg":"#ffffff","clusterBorder":"#ffffff","edgeLabelBackground":"#ffffff"},"flowchart":{"htmlLabels":true,"curve":"linear"}}}%%
flowchart TD
    classDef gray fill:#f5f5f5,stroke:#b3b3b3,color:#171717,stroke-width:1px
    classDef green fill:#edf3e8,stroke:#8a9f7a,color:#171717,stroke-width:1px
    classDef blue fill:#e8f0f9,stroke:#7f9fc4,color:#171717,stroke-width:1px
    classDef pink fill:#f5e7ec,stroke:#bd7c8f,color:#171717,stroke-width:1px

    task["1 · Task<br/>y = w·x + noise"] --> desc["2 · Description<br/>precision r, reliability p"]
    desc --> score["3 · Scoring<br/>regret over a horizon"]
    score --> ess["4 · ESS<br/>examples with equal regret"]
    ess --> exact["5 · Exact learner<br/>RQ1, RQ2"]
    ess --> net["6 · Trained networks<br/>RQ3"]
    exact --> compare["7 · Comparison<br/>network against exact"]
    net --> compare

    class task,desc gray
    class score,ess green
    class exact,net blue
    class compare pink
```

## 1. Task

Each task is a hidden weight vector $w$ with $d$ entries. An example is an
input $x$ and an answer

$$
y = w^\top x + \epsilon, \qquad x \sim \mathcal{N}(0, I_d), \quad
\epsilon \sim \mathcal{N}(0, 1).
$$

Without a description, $w \sim \mathcal{N}(0, a_0 I_d)$. The noise variance is
1 throughout, so $a_0$ is the signal-to-noise ratio of the base prior.

## 2. Description

A description is a vector $m$ with $d$ entries. It is a statement about the
weights, not text.

| Property | Symbol | Meaning |
| --- | --- | --- |
| Precision | $r$ | If the description is correct, $w \sim \mathcal{N}(m, r\,a_0 I_d)$. Smaller $r$ is more precise. |
| Reliability | $p$ | The description is correct with probability $p$. Otherwise $w \sim \mathcal{N}(0, a_0 I_d)$, unrelated to $m$. |

The vector is drawn as $m \sim \mathcal{N}(0, (1 - r)\,a_0 I_d)$, so $w$ has
the same overall distribution with or without a description.

Dimension $d$ belongs to the task, not to the description. It is reported as
part of the setting.

## 3. Scoring

A learner predicts a distribution for each answer. Its **regret** is its log
loss minus the log loss of an oracle that knows $w$, in nats. The **horizon**
$N$ is how many answers it predicts in a row, seeing each true answer before
the next prediction.

| Horizon | Corresponds to |
| --- | --- |
| $N = 1$ | One question, one answer |
| $N$ large | A long session with feedback |

The trained networks of section 6 are scored by the same log loss, as in
Genewein et al.: the network predicts a Gaussian for the answer, and its
regret is its log loss minus the oracle's, in nats.

## 4. Effective sample size

Take a learner with the description and no examples, and a learner with $n$
examples and no description. The description's **ESS** is the $n$ at which
the second learner's regret over the next $N$ predictions equals the
first's, interpolated between integers.

## 5. Exact learner (RQ1, RQ2)

No network and no prompt. The Bayes-optimal prediction is computed by
formula; the only randomness is the draw of inputs and tasks.

| | RQ1 | RQ2 |
| --- | --- | --- |
| Code | `src/descriptor_icl/gaussian.py` | `src/descriptor_icl/mixture.py` |
| Script | `results/rq1-single-query-gap/run.py` | `results/rq2-reliability/run.py` |
| Reliability | $p = 1$ | $p \in \{1, 0.99, 0.9\}$ |
| Settings $(d, a_0)$ | $d \in \{1, \dots, 64\}$, $a_0 \in \{1, 10, 100\}$ | (4, 10), (16, 10), (16, 100), (64, 10) |
| Precision $r$ | 0.1, 0.01, 0.001 | 0.1, 0.01, 0.001 |
| Seeds | 0, 1, 2 | 0, 1, 2 |

## 6. Trained networks (RQ3)

This section follows one prompt from its creation to its score. The numbers
come from `meta.sample_batch` and `mixture.py` with $d = 2$, $a_0 = 10$,
$r = 0.05$, $p = 0.9$, and generator seed 3. The experiments use $d = 5$;
the steps are the same.

**Units.** The model sees the task in the units of Garg et al. and Huang &
Ge, $w \sim \mathcal{N}(0, I)$. This is the task of sections 1 and 2 with
$m$, $w$, and every $y$ divided by $\sqrt{a_0}$, so the noise variance is
$1 / a_0 = 0.1$. All numbers in 6.1 to 6.6 are in these units.

### 6.1 Draw a task and a description

| Step | Draw | Value in the example |
| --- | --- | --- |
| Description | $m \sim \mathcal{N}(0, (1 - r) I)$ | $m = (0.74, 1.46)$ |
| Is it correct? | Yes with probability $p = 0.9$ | Yes |
| Hidden weights | Correct: $w \sim \mathcal{N}(m, r I)$ | $w = (0.54, 1.55)$ |
| Inputs | $x_k \sim \mathcal{N}(0, I)$ | $(0.42, 0.26)$, $(-2.39, -0.50)$, $(1.02, 1.05)$ |
| Answers | $y_k = w^\top x_k + \text{noise}$ | $0.25$, $-2.23$, $2.04$ |

Half the training prompts have no description; their $w$ comes from the base
prior. Prompts are drawn fresh at every step and never reused.

### 6.2 Write the prompt

A prompt is an optional description, $n$ examples, and one question. Each
row is one token of $d + 3$ numbers. There is no text and no tokenization;
the numbers enter the model through one linear layer. With $n = 2$:

```text
         is-description  is-example   vector          answer
row 0:        1              0        0.74   1.46      0.00     the description m
row 1:        0              1        0.42   0.26      0.25     example: x1 with y1
row 2:        0              1       -2.39  -0.50     -2.23     example: x2 with y2
row 3:        0              0        1.02   1.05      0.00     question: x3, answer blank
```

This is the prefix embedding of Huang & Ge (2025, their eq. 2), row for row:
two marker columns, the descriptor in the first token, each input beside its
own answer, and a final question with a blank answer.

| Variation | What changes |
| --- | --- |
| No description | Row 0 is hidden from the model |
| Fewer examples | The question moves up; later rows are hidden |

The number of examples $n$ is drawn uniformly from $0, \dots, K - 1$ for each
prompt, with $K = 16$. Hidden rows keep every prompt the same shape.

### 6.3 What is stated and what is learned

| Quantity | In the prompt? | How the model knows it |
| --- | --- | --- |
| $m$ | Yes | It reads it |
| Precision $r$ | No | Fixed per model; learned from training prompts |
| Reliability $p$ | No | Fixed per model; learned from training prompts |

### 6.4 Model and output

| Item | Value | Source |
| --- | --- | --- |
| Model | Transformer, 12 layers, 8 heads, width 256, GELU, no dropout; 9.5M parameters | Garg et al., Section 2 |
| Positions | None; the marker columns tell the rows apart | Huang & Ge |
| Output | A mean and a log variance for the answer, read at the question row | Genewein et al.; decision 19 |

### 6.5 Loss

Log loss: the negative log density of the true answer under the model's
Gaussian, as in Genewein et al. In the example the model outputs a mean and
a spread for the blank in row 3, say $2.14 \pm 0.3$, and is scored by the
log density at $2.04$. Each prompt contributes one question.

### 6.6 What the exact learner predicts for the same prompt

The exact learner's predictive is a Gaussian (stage 1) or a mixture of two
(stage 2); the table shows its mean. Under log loss the regret also counts
how well the spread is set.

| Learner | How it predicts the blank in row 3 | Prediction | Squared error against 2.04 |
| --- | --- | --- | --- |
| Oracle who knows $w$ | $w^\top x_3$ | 2.18 | 0.018 |
| Stage 1: description always right | Fit to the two examples, pulled toward $m$ | 2.14 | 0.009 |
| Stage 2: right with probability 0.9 | $0.97 \times 2.14 + 0.03 \times 0.91$ | 2.10 | 0.003 |
| No description | Fit to the two examples, pulled toward zero | 0.91 | 1.275 |

In stage 2 the weight 0.97 is the learner's belief that the description is
right. It starts at the reliability, 0.90, and rises because the description
predicted the two examples well.

**Regret** is a learner's log loss minus the oracle's. For this one
prompt the learners with a description happen to beat the oracle on the
mean, because the noise in $y_3$ fell their way. Results use the average
over 20,000 prompts, where the oracle is best.

### 6.7 From regret to ESS

Averaged over prompts at the experimental setting ($d = 5$, $a_0 = 10$), in
nats (`results/rq3-meta-trained/targets.csv`; the table below is refreshed
from that file):

| Examples, no description | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Regret | 1.87 | 1.74 | 1.58 | 1.38 | 1.13 | 0.88 | 0.68 | 0.55 | 0.45 |

| Description alone | Regret | Falls on the curve at | ESS |
| --- | --- | --- | --- |
| $r = 0.5$, $p = 1$ | 1.53 | between 2 and 3 examples | 2.24 |
| $r = 0.5$, $p = 0.9$ | 1.60 | between 1 and 2 examples | 1.85 |
| $r = 0.05$, $p = 1$ | 0.58 | between 6 and 7 examples | 6.76 |
| $r = 0.05$, $p = 0.9$ | 0.87 | between 5 and 6 examples | 5.04 |

These four ESS values are the targets for the networks. The same
calculation is done with the network's regrets in place of the exact
learner's. RQ3 asks whether the two agree.

### 6.8 Training

| Item | Value | Source |
| --- | --- | --- |
| Optimiser | Adam, learning rate $10^{-4}$, constant | Garg et al., Appendix A |
| Gradient clipping | Norm 1 | Huang & Ge |
| Batch | 1024 prompts | Ours; one question per prompt carries less signal than Garg et al.'s every-row loss at batch 64 |
| Steps | 40,000 for every model, a fixed budget. The pilot (6.11) checks that training loss has flattened by then | Garg et al. train for a fixed budget; a fixed budget avoids choosing the stopping point by the comparison being reported |
| Hardware | Colab L4, one process; TF32 matmuls and bf16 autocast for the forward pass, loss in fp32 (about 10 steps/s; fp32 gives 4) | Numerics only; the model, data, loss, and optimiser are unchanged |

### 6.9 Stages

The experiments start with the easiest task for the model and add one
difficulty at a time. A stage begins only after the previous one shows the
network reaching the exact learner's regret.

| Stage | Adds | Reliability $p$ | Exact answer the model must learn |
| --- | --- | --- | --- |
| 1 | Nothing: the description is always right | 1 | Ridge regression pulled toward $m$ |
| 2 | The description can be wrong | 0.9 | Weigh "right" against "wrong" from the examples |
| 3 | Test reliability differs from training | as stage 2 | None learned; this measures how the network's trust transfers |

### 6.10 Grid

| Factor | Values | Count | Reason for the values |
| --- | --- | --- | --- |
| Precision $r$ | 0.5, 0.05 | 2 | A coarse description that pins each weight to 71% of its typical size (worth about $d/2$ alone) and a precise one that pins it to 22% (worth more than $d$ alone, past where the $d(1-r)$ rule holds) |
| Reliability $p$ | 1 (stage 1), 0.9 (stage 2) | 2 | The reliable case, and the 10% vague weight of Schmidli et al.'s robust-prior example (RBesT's default is 20%); on the RQ2 grid, where the cap is visible but not extreme |
| Seed | 0 | 1 | One model per configuration (decision 20) |
| Setting | $d = 5$, $a_0 = 10$, $K = 16$ | 1 | $d$ from Huang & Ge; $K$ covers the largest ESS, 6.8, twice over |
| Prompts with a description | Half | | Equal practice with and without |

4 models, 2 per stage. Each model handles every number of examples from
0 to 15; there is not one model per $n$.

### 6.11 Pilot

One stage-1 model ($p = 1$, $r = 0.05$, seed 0) is trained for 40k steps
with a snapshot every 5k. The training loss, on fresh prompts every step,
must have flattened by 40k; that confirms the budget. Each snapshot's gap
to the exact learner is recorded in `pilot.csv` as evidence of convergence,
not as a stopping rule: stopping a model when it looks most Bayes-optimal
would select the result being reported. The pilot is the grid's $p = 1$,
$r = 0.05$ model. If the gap does not close, the design is revisited.

An earlier pilot with a distribution output and log loss (6 layers, width
128) came within 0.03 to 0.05 nats of the exact learner after 10,000 steps.
It shows the task is learnable; it is not part of the results.

## 7. Comparison

The trained network and the exact learner receive the same fresh prompts
(`results/rq3-meta-trained/evaluate.py`). The network is run once for each
number of examples.

| Readout | Question |
| --- | --- |
| Regret after $n$ examples | Does the network predict as well as the exact learner? |
| ESS | Is the description worth as many examples to the network? |
| Implied trust | Which trust $q$ makes the exact learner's prediction closest to the network's? |
| Shifted reliability (`--p-test`) | What happens when test reliability differs from training? |

## Grounding

Every element either matches a published setup or is required by the
question. Elements that are neither are listed under Departures.

| Design element | Source |
| --- | --- |
| In-context linear regression, $x \sim \mathcal{N}(0, I)$, $w \sim \mathcal{N}(0, I)$ | Garg et al. 2022; Huang & Ge 2025 |
| Model size, optimiser, learning rate | Garg et al. 2022 |
| Gaussian output and log loss against the exact learner | Genewein et al. 2025 |
| Prefix layout, marker columns, no positions, one question per prompt, $d = 5$, gradient clipping | Huang & Ge 2025 |
| One model for every number of examples | Garg et al. 2022 |
| Robust mixture prior for reliability | Schmidli et al. 2014 |
| ESS by matching a prior against observations | Morita et al. 2008; Reimherr et al. 2021 |

## Departures

| Piece | Related work | Ours | Reason |
| --- | --- | --- | --- |
| What the descriptor describes | The mean of the inputs | The weights | Required: it must carry information about the task to have a worth in examples |
| Reliability | None | Right with probability $p$ | Required: second axis of the study |
| Noise in the answers | None | Variance 0.1 | Required: without noise, $d$ examples determine $w$ exactly, so no description could be worth more than $d = 5$ examples; the precise description is worth 6.8 |
| Attention | Linear (Huang & Ge) | Standard (Garg et al.) | Required: weighing two hypotheses is not linear in the examples |
| With and without a descriptor | Separate models (Huang & Ge) | One model; the description row is present or hidden | Kept by Bruce. The ESS then compares a learner with itself |
| Number of examples | Fixed (Huang & Ge); every row scored (Garg et al.) | One question after a random number of examples | Kept by Bruce. The ESS needs regret at every number of examples |
| Examples per prompt | 40 (Garg et al.), 50 (Huang & Ge) | Up to 15 | Kept by Bruce. Covers the largest ESS twice over |
| Seeds | 5 (Huang & Ge) | 1 | Bruce, 2026-09-30: one training per configuration; the exact learner's comparison does not need a spread across seeds |

## Decisions

| # | Decision | Date | Reason |
| --- | --- | --- | --- |
| 1 | The paper has two axes, precision and reliability. Dimension is part of the setting. | 2026-09-29 | Dimension is a property of the task and has no clear counterpart in real prompts. |
| 2 | A description is a statement about the weights. | 2026-09-29 | A description of the inputs is worth zero examples to the exact learner. |
| 3 | The description occupies the first token only. | 2026-09-29 | Huang & Ge's prefix embedding. |
| 4 | The description states only $m$. Precision is fixed per model. | 2026-09-29 | Nothing for the model to misread; same treatment as reliability. |
| 5 | One question per prompt, after a random number of examples. | 2026-09-29 | Huang & Ge's loss form. |
| 6 | Answers are noisy. | 2026-09-29 | See Departures. |
| 7 | Each input sits beside its own answer, and the prompt ends with a question row. | 2026-09-29 | Huang & Ge's layout exactly. |
| 8 | A prompt without a description hides the description row. | 2026-09-29 | A prompt either has an instruction or does not; no extra field. |
| 9 | No position information and no causal mask. | 2026-09-29 | Huang & Ge; examples have no order. |
| 10 | $d = 5$ and at most 15 examples. | 2026-09-29 | Huang & Ge's dimension. |
| 11 | Experiments run in stages, easiest first (6.9). | 2026-09-29 | Bruce: keep it easy for the model and raise the difficulty later. |
| 12 | Unreliable descriptions are studied at $p = 0.9$ only. | 2026-09-29 | Bruce: one level is enough to start. |
| 13 | One model covers all numbers of examples. | 2026-09-29 | The ESS compares regrets across $n$ for one learner; Garg et al. train the same way. |
| 14 | **Standard:** match related work wherever possible; depart only where the question requires it, and record the reason under Departures. | 2026-09-29 | Bruce: every choice must be justifiable, and each departure invites a question. |
| 15 | The networks output one number and train on squared error. Supersedes the distribution output and log loss. | 2026-09-29 | Decision 14. Garg et al. and Huang & Ge; squared error can answer the question. |
| 16 | Model size of Garg et al.: 12 layers, 8 heads, width 256. Supersedes 6 layers, width 128. | 2026-09-29 | Decision 14. |
| 17 | Adam at a constant learning rate of $10^{-4}$. Supersedes warm-up and decay at 3e-4. | 2026-09-29 | Decision 14. Garg et al. |
| 18 | The model sees $w \sim \mathcal{N}(0, I)$. Supersedes weights of variance 10. | 2026-09-29 | Decision 14. The same task in the units of Garg et al. and Huang & Ge. |
| 19 | The networks output a mean and a log variance and train on log loss. Supersedes decision 15. | 2026-09-30 | Bruce. One ESS definition through the paper; Genewein et al. is the evaluation the paper follows; the earlier log-loss pilot learned the task. |
| 20 | One seed per configuration. Supersedes three seeds. | 2026-09-30 | Bruce. Compute; each model is compared with the exact learner on 20,000 prompts, so the comparison itself has small error. |
| 21 | Grids are powers of ten: precision $r \in \{0.1, 0.01, 0.001\}$ (a description worth 1, 10, 100 examples in the long run at $a_0 = 10$), reliability $p \in \{1, 0.99, 0.9\}$ (wrong never, one time in 100, one time in 10). Supersedes the five-value grids of section 5. | 2026-09-30 | Bruce: pick the ends that matter, a factor of 10 apart, instead of many nearby values. |
| 22 | Examples in hand are reported at $n \in \{0, 1, 10, 100\}$: alone, after identification, below the dimension, verified. | 2026-10-01 | Bruce. Each value is a different regime of Proposition 3. |
| 23 | Networks: prompts hold up to $20 = 4d$ examples and the models cover $r \in \{0.1, 0.01\}$. Supersedes decision 10's 15 examples and the $r \in \{0.5, 0.05\}$ models. | 2026-09-30 | Decision 21. A description worth 100 examples has no crossing within a 20-example prompt at $d = 5$, so the networks take the two ends they can resolve; 20 examples cover the $r = 0.01$ reliable ESS of 15.6. |

## Open questions

None.

## Deferred

- The input-mean descriptor of Huang & Ge as a second condition.
- An LLM experiment.
- An LSTM, which the proposal names.
