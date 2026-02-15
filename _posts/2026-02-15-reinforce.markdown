---
layout: post
title:  "The REINFORCE Trick: Optimizing What You Can't Differentiate"
date:   2026-02-15 12:00:00 -0700
categories: RL for LLMs
---

*This is the first post in a new series called "RL for Language Models." The previous series, ["ML Principles for Practitioners,"]({% post_url 2026-02-08-logistic-regression-loss-from-bradley-terry %}) built from the Bradley-Terry model to [DPO]({% post_url 2026-02-11-dpo %}), which sidesteps reinforcement learning entirely. This series takes the other path: what happens when we actually run RL on a language model? It starts with understanding how to take a gradient when the output is sampled.*

---

In supervised learning, training is straightforward. You have inputs, you have correct outputs, and you have a loss function that measures how far off your predictions are. You compute the gradient of the loss and update the parameters. Every step of this process is differentiable.

But the most capable language models today aren't trained only with supervised learning. They're also trained with reinforcement learning: the model generates a response, a reward signal scores it (maybe a human rating, maybe a test case passing, maybe a proof checking out), and the model learns from that score. No one tells the model *what* the right response is. It only learns *how good* its response was.

This creates a fundamental problem. In supervised learning, you can differentiate the loss directly through the model's output. In RL, the model *samples* its output -- it rolls dice at each token position to decide which token to produce -- and you can't differentiate through dice rolls. The output is discrete, stochastic, and the connection between parameters and reward passes through a sampling operation that has no gradient.

The solution is a single mathematical identity from 1992, known as the REINFORCE trick. It's the foundation of all RL training for language models -- PPO, GRPO, and everything in between. This post derives it from scratch, explains what it means, and shows why it works but also why it's noisy.

## The goal: maximize expected reward

Let's start by writing down what we're trying to do. Suppose we have a language model $\pi_\theta$ that generates responses to prompts. For a given prompt, the model can produce many different responses, each with some probability. We have a reward function $R(y)$ that scores each response (higher is better).

We want to find parameters $\theta$ that make the model produce high-reward responses. Formally, we want to maximize the **expected reward**:

$$J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[R(y)] = \sum_y \pi_\theta(y) \cdot R(y)$$

This is a weighted average: the reward of each possible response, weighted by how likely the model is to generate it. We want the model to put high probability on high-reward responses and low probability on low-reward ones.

**A tiny example to keep things concrete.** Suppose the model can only output one of three words: "yes", "no", or "maybe." The current model assigns probability 0.3 to "yes" (reward 8), probability 0.5 to "no" (reward 3), and probability 0.2 to "maybe" (reward 5).

The expected reward is $0.3 \times 8 + 0.5 \times 3 + 0.2 \times 5 = 2.4 + 1.5 + 1.0 = 4.9$. The model spends most of its probability on "no," which is the worst response. We'd like to shift probability toward "yes" (reward 8). That would increase the expected reward.

In supervised learning, we'd just say "the right answer is 'yes'" and minimize $-\log \pi_\theta(\text{yes})$. But in the RL setting, no one tells us the right answer. We only know the rewards after the model produces a response.

## Why the gradient is hard to compute

To maximize $J(\theta)$ with gradient ascent, we need its gradient $\nabla_\theta J(\theta)$. Let's try the obvious thing and differentiate:

$$\nabla_\theta J(\theta) = \nabla_\theta \sum_y \pi_\theta(y) \cdot R(y) = \sum_y R(y) \cdot \nabla_\theta \pi_\theta(y)$$

The reward $R(y)$ doesn't depend on $\theta$ (it's a fixed score for each response), so it passes through the gradient. This formula is exact. But we can't compute it, for two reasons.

**Problem 1: the sum is intractable.** For a language model, $y$ ranges over all possible token sequences. Even for a short response of 100 tokens with a vocabulary of 50,000, the number of possible sequences is astronomical. We cannot enumerate them.

**Problem 2: we can't estimate it by sampling in its current form.** The natural fix for an intractable sum is Monte Carlo estimation: sample a few $y$'s and average. We can easily sample from $\pi_\theta$ -- that's just running the model. But look at the sum we need to estimate: each term is $R(y) \cdot \nabla_\theta \pi_\theta(y)$.

To estimate a sum by sampling from a distribution $\pi_\theta$, we need the sum to have the form $\sum_y \pi_\theta(y) \cdot [\text{something}]$, because then the recipe is simple: sample $y \sim \pi_\theta$, compute [something] for each sample, and average. That gives an unbiased estimate of the sum.

Our sum doesn't have this form. It has $\nabla_\theta \pi_\theta(y)$ instead of $\pi_\theta(y)$ as the weight. We need to massage the expression to put $\pi_\theta(y)$ out front.

## The REINFORCE trick

The idea is to force $\pi_\theta(y)$ to appear as a weight so we can estimate the gradient by sampling. We do this by multiplying and dividing by $\pi_\theta(y)$ inside the sum:

$$\sum_y R(y) \cdot \nabla_\theta \pi_\theta(y) = \sum_y \pi_\theta(y) \cdot R(y) \cdot \frac{\nabla_\theta \pi_\theta(y)}{\pi_\theta(y)}$$

Now $\pi_\theta(y)$ sits out front as a weight, which is what we needed. The remaining piece is the ratio $\frac{\nabla_\theta \pi_\theta(y)}{\pi_\theta(y)}$. A standard identity from calculus simplifies this. The chain rule applied to $\log$ says: the derivative of $\log f$ is $f'/f$. Reading this in the other direction: $f'/f = (\log f)'$. So:

$$\frac{\nabla_\theta \pi_\theta(y)}{\pi_\theta(y)} = \nabla_\theta \log \pi_\theta(y)$$

That's the entire mathematical content of the "trick." Substituting back, the gradient of the expected reward becomes:

$$\nabla_\theta J(\theta) = \sum_y \pi_\theta(y) \cdot R(y) \cdot \nabla_\theta \log \pi_\theta(y) = \mathbb{E}_{y \sim \pi_\theta}\left[R(y) \cdot \nabla_\theta \log \pi_\theta(y)\right]$$

This is the **REINFORCE gradient estimator** (Williams, 1992). Let's compare what we had before and after. Before: a sum weighted by $\nabla_\theta \pi_\theta(y)$, which we couldn't estimate by sampling. After: an expectation under $\pi_\theta$, which we *can* estimate by sampling. The mathematical content was just two moves: multiply-and-divide by $\pi_\theta$, then recognize $f'/f$ as $(\log f)'$.

Now we have a recipe. Sample responses from the model, compute the thing inside the expectation for each sample, and average. That gives us a gradient estimate we can use for optimization.

**The algorithm.** To estimate the gradient and update the model:

1. Sample $K$ responses from the current model: $y_1, \ldots, y_K \sim \pi_\theta$
2. Score each one: $R(y_1), \ldots, R(y_K)$
3. Estimate the gradient: $\hat{g} = \frac{1}{K} \sum_{k=1}^{K} R(y_k) \cdot \nabla_\theta \log \pi_\theta(y_k)$
4. Update the parameters: $\theta \leftarrow \theta + \alpha \hat{g}$

Sample, score, compute the weighted gradient, update. That's REINFORCE.

## What REINFORCE actually tells the model to do

The formula $R(y) \cdot \nabla_\theta \log \pi_\theta(y)$ has a clean interpretation. The term $\nabla_\theta \log \pi_\theta(y)$ is the gradient that would *increase* the log-probability of response $y$. It points in the direction in parameter space that makes the model more likely to produce $y$. This is the same gradient you'd use in supervised learning if someone told you "$y$ is the correct answer."

REINFORCE says: follow this gradient, but *scale it by the reward*.

- If $R(y)$ is large and positive: take a big step toward making $y$ more likely. This response scored well; produce it more often.
- If $R(y)$ is small and positive: take a small step. The response was OK but not great.
- If $R(y)$ is negative: step in the *opposite* direction -- make $y$ *less* likely. This response was bad; avoid it.

**In plain English: REINFORCE tells the model to repeat what worked and avoid what didn't, in proportion to how well or poorly each response scored.** If you're learning to cook and a dish gets rave reviews, you'd make that dish more often. If it gets mediocre reviews, you'd adjust a little. If it's terrible, you'd stop making it. REINFORCE does exactly this, with the reward as the review.

**Back to our example.** Suppose we sample one response and get "yes" (reward 8). The gradient estimate is $8 \cdot \nabla_\theta \log \pi_\theta(\text{yes})$: a strong push to make "yes" more probable. Good -- "yes" is the best response.

But suppose instead we sample "no" (reward 3). The estimate is $3 \cdot \nabla_\theta \log \pi_\theta(\text{no})$: a weaker but still positive push to make "no" more probable. The model takes a step toward generating "no" more often.

But "no" is the *worst* response. Why would we want to make it more probable? Because $R(\text{no}) = 3 > 0$, so REINFORCE reads this as "that was good, do more of it." With only one sample, the model has no way to know that "yes" would have scored 8. All it sees is that "no" scored 3, which is positive, so it nudges up.

Over many samples, the expected gradient is correct. Even though "no" gets sampled more often (probability 0.5 vs 0.3), "yes" contributes more to the expected gradient because its reward is higher ($0.3 \times 8 = 2.4$ vs $0.5 \times 3 = 1.5$). The expected gradient points toward "yes." But any *single* gradient estimate can be pointing the wrong way.

## The variance problem

This is the central challenge of REINFORCE.

The gradient estimator is **unbiased**: averaged over many samples, it equals the true gradient. But any single estimate can be far from the truth. Which direction the gradient points, and how large it is, depend entirely on which response we happened to sample. That randomness is called **variance**, and high variance means slow, noisy learning.

The problem is especially bad when all rewards are positive, which is common in practice. If every response scores between 3 and 8, then REINFORCE pushes *every* sampled response up. It never pushes anything *down*. The net effect over many samples is correct (high-reward responses get pushed up more), but each individual update is an upward nudge toward whatever happened to be sampled, regardless of whether it was the best or worst option.

**An analogy.** Suppose you're trying to figure out which of three restaurants is best. You visit one at random each night and rate the meal. All three are decent (ratings 6, 7, and 8 out of 10). After each visit, you tell yourself "that was good, I should go there more." You're never telling yourself "that was bad, I should go there less." Eventually the differences will emerge -- you'll visit the 8-rated restaurant slightly more often -- but it takes many meals because every night feels positive.

Now imagine the ratings were $-2$, $0$, and $+3$. You'd get a clear "avoid this" signal on some nights and a clear "go here" signal on others. You'd figure out the ranking much faster. The absolute quality hasn't changed -- the same restaurant is still the best. But the *relative* signals are sharper when some rewards are negative.

This is the key insight: REINFORCE learns from *absolute* rewards, but what matters for learning is the *relative* difference between good and bad options. We need a way to turn absolute rewards into relative ones.

## Baselines: a simple fix with a deep effect

The fix is elegant. Instead of weighting the gradient by the raw reward $R(y)$, weight it by $R(y) - b$, where $b$ is a constant called the **baseline**:

$$\hat{g} = \frac{1}{K} \sum_{k=1}^{K} \left(R(y_k) - b\right) \cdot \nabla_\theta \log \pi_\theta(y_k)$$

The baseline shifts the rewards so that below-average responses get negative weight and above-average ones get positive weight. Now REINFORCE explicitly pushes *down* the probability of below-average responses and pushes *up* the probability of above-average ones. The "avoid this" signal that was missing before is present in every update.

**Why this doesn't break anything.** You might worry that changing the reward changes the expected gradient. It doesn't. Here's the one-line proof. The bias introduced by subtracting $b$ would be:

$$b \cdot \mathbb{E}_{y \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(y)\right] = b \cdot \nabla_\theta \underbrace{\sum_y \pi_\theta(y)}_{= \, 1} = b \cdot \nabla_\theta 1 = 0$$

The key step: probabilities always sum to 1. That's true for any value of $\theta$. So $\sum_y \pi_\theta(y) = 1$ is a constant with respect to $\theta$, and the gradient of a constant is zero. Subtracting *any* constant from the reward introduces zero bias. We're completely free to choose whatever $b$ reduces variance the most.

**Back to our example.** The rewards are 8 ("yes"), 3 ("no"), and 5 ("maybe"). A natural baseline is the mean reward: $b \approx 5.3$. Subtracting it gives adjusted rewards of $+2.7$ ("yes"), $-2.3$ ("no"), and $-0.3$ ("maybe").

Now if we sample "no," the weight is $-2.3$: the gradient pushes the probability of "no" *down*. If we sample "yes," the weight is $+2.7$: it pushes "yes" *up*. If we sample "maybe," the weight is $-0.3$: a small push down. Every sample now carries a directional signal: better than average, or worse than average. Compare this to before, where every sample pushed up.

**What's the best baseline?** Theory says the variance-minimizing baseline is the expected reward itself: $b^* = \mathbb{E}_{y \sim \pi_\theta}[R(y)]$. We don't know this exactly (estimating it is part of what we're trying to do), but we can approximate it by averaging the rewards from the current batch of samples. More sophisticated approaches go further: they *learn* a function that predicts the expected reward for each specific prompt. That learned, input-dependent baseline is called a **critic** or **value function**, and it leads directly to PPO. We'll cover it in the next post.

The quantity $R(y) - b$ has a name in RL: the **advantage**. It measures how much better (or worse) a specific response is compared to what you'd expect on average. Positive advantage means "better than expected," negative means "worse than expected." This concept will appear throughout this series.

## Connection to language models

Everything so far used a one-word example. Real language models generate sequences of tokens, one at a time. Here's how REINFORCE extends to that setting.

A language model generates a response $y = (t_1, t_2, \ldots, t_T)$ autoregressively. The probability of the full response factors into per-token conditional probabilities:

$$\pi_\theta(y) = \prod_{k=1}^{T} \pi_\theta(t_k \mid t_{<k})$$

Taking the log turns the product into a sum (the same product-to-sum-via-log pattern from [Post 2]({% post_url 2026-02-09-what-is-maximum-likelihood-really %}) of the ML Principles series):

$$\log \pi_\theta(y) = \sum_{k=1}^{T} \log \pi_\theta(t_k \mid t_{<k})$$

So the REINFORCE gradient for a single sampled response is:

$$R(y) \cdot \nabla_\theta \log \pi_\theta(y) = R(y) \cdot \sum_{k=1}^{T} \nabla_\theta \log \pi_\theta(t_k \mid t_{<k})$$

Look at what this says. The *same* reward $R(y)$ -- a single number for the entire response -- multiplies the gradient at *every* token position. Token 1 gets the same credit as token 50 and token 200, regardless of which tokens actually made the response good or bad.

**This is the credit assignment problem.** Suppose the model generates a 200-token explanation of photosynthesis. The first 150 tokens are clear and accurate. Token 151 introduces a factual error. The remaining tokens are fine but follow from the error. A human rates the whole response poorly.

REINFORCE penalizes every token equally. Tokens 1 through 150, which were excellent, get the same negative gradient as token 151, which caused the problem. The model might learn to avoid *everything* about this response, including the good parts.

Supervised fine-tuning doesn't have this problem. There, each token gets its own gradient from its own target: token 47 is responsible for predicting token 47, not for the quality of the whole sequence. The training signal is local.

A constant baseline (like the mean reward across a batch) helps with the overall scale -- it centers rewards around zero so that bad responses get negative weight. But it doesn't help with credit assignment. Every token in a bad response still gets the same penalty, even if most tokens were good.

What we really want is a *per-token* baseline: at each position $k$, an estimate of "given the text generated so far, how good is the rest of the response going to be?" That would let each token's gradient reflect its own marginal contribution: did *this specific token* make the response better or worse than expected from this point on?

Learning such a position-dependent baseline is exactly what the **critic** (also called the value function) does in PPO, and that's the topic of the next post.

## What's next

Let's take stock. We started with a fundamental gap: supervised learning can differentiate the loss directly through the model's output, but RL can't, because the model's output is sampled. The REINFORCE trick bridges this gap with a single algebraic manipulation (multiply and divide by $\pi_\theta$, recognize $f'/f$ as $(\log f)'$), converting an intractable sum into an expectation we can estimate by sampling.

The result is clean but noisy. Baselines reduce the noise by centering rewards, turning every sample into a clear "better than average" or "worse than average" signal. But even with baselines, REINFORCE treats the entire response as one atomic action. It assigns the same reward to every token and can't tell which ones actually mattered.

The next post covers **PPO** (Proximal Policy Optimization), the algorithm that was used to train InstructGPT and ChatGPT. PPO builds on REINFORCE with two additions: a learned per-token baseline (the critic) that addresses credit assignment, and a clipping mechanism that prevents destructively large updates. These additions make PPO much more stable than raw REINFORCE, but they come at a steep cost: four full-sized models must be loaded into GPU memory simultaneously. Understanding that cost is what makes the final post's topic, GRPO, so compelling.

---

**References**

1. R. J. Williams. ["Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning."](https://link.springer.com/article/10.1007/BF00992696) *Machine Learning*, 1992. The original REINFORCE paper.
2. R. S. Sutton and A. G. Barto. [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book.html), Chapter 13. Policy gradient methods, the REINFORCE algorithm, and variance reduction through baselines.
3. N. Lambert. [RLHF Book, Chapter 9: Policy Gradient Methods.](https://rlhfbook.com/c/09-policy-gradients) Covers REINFORCE and its application to language model training.
4. [Post 2 in the ML Principles series]({% post_url 2026-02-09-what-is-maximum-likelihood-really %}). The product-to-sum-via-log pattern that reappears in the autoregressive factorization.
