---
layout: post
title:  "Where Does the Logistic Loss Come From?"
date:   2026-02-08 12:00:00 -0700
categories: ML Principles
---

*This is the first post in a series called "ML Principles for Practitioners." Each post takes a common ML tool and explains where it comes from and when to trust it.*

---

If you've trained a binary classifier, you've used the logistic loss (also called binary cross-entropy). It's the default in every ML framework. But why *this* formula? Why not squared error, or something else entirely?

The answer is surprisingly clean. The logistic loss comes from a simple model of pairwise comparison called the **Bradley-Terry model**. Understanding this connection gives you a principled reason to use the loss, and tells you when it might be the wrong choice.

## The problem: ranking items from comparisons

Imagine you have a set of items (basketball teams, AI chatbots, search results) and all you observe is pairwise outcomes. Team A beat Team B. Chatbot X was preferred over Chatbot Y. You want to assign a strength score to each item.

The naive approach is to count wins. But that fails when matchups are uneven. A team that only plays weak opponents will look better than a team that plays tough ones. We need a model that accounts for *who you played*, not just whether you won.

## The Bradley-Terry model

The idea is simple. Give every item a score on a number line. The further apart two items are, the more confident we are that the stronger one wins.

Formally, item $i$ gets a score $s_i$. The probability that $i$ beats $j$ is:

$$P(i \text{ beats } j) = \frac{e^{s_i}}{e^{s_i} + e^{s_j}}$$

With a bit of algebra (divide top and bottom by $e^{s_i}$), this simplifies to:

$$P(i \text{ beats } j) = \frac{1}{1 + e^{-(s_i - s_j)}} = \sigma(s_i - s_j)$$

That's the **sigmoid function** applied to the score difference. The probability depends only on how much stronger $i$ is than $j$, not on their absolute scores. You could add 100 to every score and nothing would change.

This model was proposed by Bradley and Terry in 1952, though Ernst Zermelo had the same idea in 1929. It keeps getting rediscovered because it's natural. If you want a simple, consistent way to turn score differences into probabilities, the sigmoid is essentially what you get.

## Why the sigmoid? Where the noise comes from

Here's one way to see why. Suppose each item has a true quality $q_i$, but when a human judge evaluates a matchup, their judgment is noisy. They perceive item $i$ as having quality $q_i + \epsilon_i$, where $\epsilon_i$ is random error. The judge picks $i$ over $j$ when:

$$q_i + \epsilon_i > q_j + \epsilon_j$$

Rearranging, $i$ wins when the noise difference $\epsilon_j - \epsilon_i$ is smaller than the quality gap $q_i - q_j$. So everything reduces to one question: what is the probability that the noise difference lands below the quality gap?

That's what a CDF answers. The CDF of a distribution tells you "what is the probability a random draw lands below some threshold?" Here the random draw is the noise difference, and the threshold is the quality gap.

Now, the **logistic distribution** is a bell-shaped distribution similar to the Gaussian. It has two parameters: a center $\mu$ and a scale $s$. We assume the noise difference is centered at zero ($\mu = 0$, meaning the noise doesn't systematically favor either item). And its CDF turns out to be:

$$F(x) = \frac{1}{1 + e^{-x/s}}$$

What about the scale $s$? We can set it to 1 without losing anything. The quality scores are parameters we're going to fit anyway, and scaling all scores by $1/s$ has the same effect as changing the noise scale. The model can only recover quality differences relative to the noise level, so we fix $s = 1$ by convention.

With $\mu = 0$ and $s = 1$, the CDF is just:

$$F(x) = \frac{1}{1 + e^{-x}} = \sigma(x)$$

The sigmoid *is* the CDF of the standard logistic distribution. So the probability that $i$ beats $j$ is simply $\sigma(q_i - q_j)$: evaluate the CDF at the quality gap. No extra constants, no complicated formula. That's where the sigmoid in Bradley-Terry comes from.

For a more formal treatment, see [Sun et al. 2024](https://arxiv.org/abs/2411.04991).

What if the noise were Gaussian instead of logistic? Then you'd use the Gaussian CDF instead of the sigmoid, giving you a **probit model**. In practice the two CDFs are nearly identical in shape, so it rarely matters which you pick. The logistic version is preferred because its CDF has a clean closed form.

## From model to loss function

Now suppose we observe $n$ matchups. For each matchup $m$, we know the two competitors ($a_m$ and $b_m$) and the outcome ($y_m = 1$ if $a_m$ won, $0$ if $b_m$ won). We want to find the scores that make our observed outcomes most likely. This is **maximum likelihood estimation**.

The probability of a single outcome is:

$$P(y_m) = \sigma(s_{a_m} - s_{b_m})^{y_m} \cdot (1 - \sigma(s_{a_m} - s_{b_m}))^{1 - y_m}$$

When $y_m = 1$ (item $a$ won), this is $\sigma(s_a - s_b)$. When $y_m = 0$ (item $b$ won), this is $1 - \sigma(s_a - s_b)$. Exactly what we'd expect.

Assuming independence, the likelihood of all $n$ outcomes is the product:

$$L = \prod_{m=1}^{n} \sigma(s_{a_m} - s_{b_m})^{y_m} \cdot (1 - \sigma(s_{a_m} - s_{b_m}))^{1 - y_m}$$

Products of many small numbers are numerically unstable, so we take the logarithm (which turns products into sums). Then we negate it, because optimizers minimize by convention. The result is:

$$\mathcal{L} = -\sum_{m=1}^{n} \left[ y_m \log \sigma(s_{a_m} - s_{b_m}) + (1 - y_m) \log(1 - \sigma(s_{a_m} - s_{b_m})) \right]$$

**This is the binary cross-entropy loss.** The logistic loss that every ML framework provides is simply asking: *find the parameters that make the observed outcomes most probable under the Bradley-Terry model.*

## Connection to binary classification

In standard logistic regression for classification, we model $P(y = 1 \mid x) = \sigma(w \cdot x)$. The "score difference" from Bradley-Terry becomes the dot product $w \cdot x$. We're essentially asking: given these features, how much does the evidence favor class 1 over class 0?

The loss function is identical. So every time you train a logistic regression classifier, you're doing maximum likelihood estimation under a Bradley-Terry-style comparison model. Class 1 competes against class 0, and the score difference is $w \cdot x$.

## Assumptions and when they break

Knowing the model behind the loss tells you when to trust it. Here are the key assumptions practitioners should watch for:

**Outcomes depend only on the score difference.** If the real relationship is more complex (say, one team only loses at high altitude), basic logistic regression won't capture it. Fix: add richer features or use a nonlinear model.

**Each observation is independent.** If outcomes are correlated (win streaks, repeated measures, sequential decisions), the model's confidence estimates will be miscalibrated. Fix: model the correlation structure explicitly.

**Transitivity.** If A beats B and B beats C, the model assumes A beats C. This breaks in rock-paper-scissors situations. Watch out in recommendation systems or games with non-transitive matchup dynamics.

**Strengths are fixed.** The model assumes items don't change over the observation period. If a team improves mid-season or a chatbot gets updated, you need time-varying extensions like decayed weighting or online updates.

**No presentation order bias.** The model assumes the noise is unbiased, meaning there's no systematic preference for whichever item is shown first, on the left, or in a particular position. In practice this bias is very common in A/B tests and RLHF annotation. Fix: add an intercept term to the score difference to capture the positional bias (this is exactly what the Stanford lecture does for home-court advantage in basketball).

**All judges have the same noise level.** The model assumes every annotator is equally noisy. In reality, some are experts and some are not. Pooling all annotations without accounting for this can distort the estimated scores. Fix: weight annotators by reliability, or model per-annotator noise explicitly.

**The logistic shape itself.** The sigmoid could be the wrong curve. The main alternative is the probit (Gaussian CDF). In practice they produce nearly identical results, so this assumption is the least likely to cause problems.

## Where you see this today

The Bradley-Terry model is quietly everywhere in modern ML:

- **RLHF reward models.** When labs like OpenAI or Anthropic train a reward model from human preferences ("response A is better than response B"), the loss function is exactly the Bradley-Terry negative log-likelihood. See the [RLHF Book](https://rlhfbook.com/c/07-reward-models) for details.

- **Chatbot Arena and Elo ratings.** The [Chatbot Arena](https://lmarena.ai/) leaderboard fits a Bradley-Terry model via logistic regression to rank LLMs from millions of human votes.

- **Any binary classification.** Every time you use `nn.BCEWithLogitsLoss` in PyTorch or equivalent in other frameworks, you are implicitly assuming a Bradley-Terry model.

The logistic loss is not an arbitrary choice. It is the principled maximum likelihood estimator under a simple, well-understood model of pairwise comparison. Knowing this helps you choose the right loss, interpret the predicted probabilities, and recognize when the model might be wrong.

---

**References**

1. R. A. Bradley and M. E. Terry. "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons." *Biometrika*, 1952. Formalized in [Stanford STATS 200 Lecture 24](https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture24.pdf).
2. H. Sun, Y. Shen, and J.-F. Ton. ["Rethinking Bradley-Terry Models in Preference-Based Reward Modeling."](https://arxiv.org/abs/2411.04991) 2024. Rigorous treatment of assumptions behind BT in the RLHF setting.
3. N. Lambert. [RLHF Book, Chapter 7: Reward Models.](https://rlhfbook.com/c/07-reward-models) Covers the modern application of Bradley-Terry to language model alignment.
