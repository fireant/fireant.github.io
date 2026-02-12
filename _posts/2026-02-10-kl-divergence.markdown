---
layout: post
title:  "KL Divergence: The One Concept You Need for Modern ML"
date:   2026-02-10 12:00:00 -0700
categories: ML Principles
---

*This is the third post in a series called "ML Principles for Practitioners." The [first post]({% post_url 2026-02-08-logistic-regression-loss-from-bradley-terry %}) derived the logistic loss from the Bradley-Terry model. The [second post]({% post_url 2026-02-09-what-is-maximum-likelihood-really %}) explained maximum likelihood estimation and mentioned that MLE is secretly minimizing something called KL divergence. This post unpacks that claim.*

---

Post 2 made a bold statement: every loss function is the negative log-likelihood of some probabilistic model. Pick a model, take the negative log, and you get a loss. We saw this for binary cross-entropy (Bradley-Terry) and for the Gaussian (MSE). But we moved through the Gaussian example quickly. Let's slow down and make it concrete, because it leads directly to the main topic of this post.

## Where MSE really comes from

Suppose you're predicting house prices. Your model takes in features (square footage, location, etc.) and predicts a price $f(x_i)$ for house $i$. The actual price is $y_i$. Your prediction won't be perfect, so there's an error: $y_i - f(x_i)$.

Now here's the modeling choice most people make without thinking about it. You assume those errors follow a **Gaussian distribution** centered at zero:

$$y_i = f(x_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

This says: the true price is your prediction plus some random noise, and that noise is bell-shaped and symmetric. Let's follow Post 2's recipe and see what loss function this gives us.

**The probability of one house's price**, given your model and the Gaussian assumption, is:

$$P(y_i \mid x_i) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - f(x_i))^2}{2\sigma^2}\right)$$

This is just the Gaussian density evaluated at the observed error. The closer $y_i$ is to $f(x_i)$, the higher the probability.

**The probability of all $n$ houses together.** We assume each house's error is independent (the error on house 3 doesn't affect the error on house 7). Independence means the joint probability is the product:

$$L = \prod_{i=1}^{n} P(y_i \mid x_i)$$

**Take the negative log** (Post 2's recipe: turn likelihood into a loss). The log turns the product into a sum:

$$-\log L = \sum_{i=1}^{n} \frac{(y_i - f(x_i))^2}{2\sigma^2} + \frac{n}{2}\log(2\pi\sigma^2)$$

The second term is a constant (it doesn't depend on $f$). So minimizing this loss over $f$ is the same as minimizing:

$$\sum_{i=1}^{n} (y_i - f(x_i))^2$$

That's **mean squared error**. The sum comes from two things: the independence assumption (which gave us the product) and the log (which turned the product into a sum). The square comes from the Gaussian. If we had assumed a different noise distribution, we'd get a different loss.

**What the Gaussian assumption actually claims about your data.** It says the prediction errors are: symmetric (equally likely to be \$10k too high or \$10k too low), bell-shaped (small errors are common, large errors are rare), and unbounded (in principle, any error size is possible, just increasingly unlikely).

**When is this wrong?** Often. House prices can't be negative, so errors are bounded on one side. Financial returns have fat tails, meaning extreme errors are much more common than a Gaussian predicts. Count data (number of clicks, number of defects) is discrete and non-negative. If you use MSE for these problems, you're still implicitly claiming Gaussian errors. It might work fine in practice, but it's worth knowing what you're assuming.

This raises a natural question: when the Gaussian assumption is wrong (and it usually is, at least a little), what exactly does MLE do? Post 2 mentioned White's theorem: MLE still finds the "closest" model to the truth. But closest in what sense? The answer is: closest in **KL divergence**. That's the concept we need to define.

## What KL divergence measures

KL divergence measures how different two probability distributions are. The formula for discrete distributions is:

$$KL(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

where $P$ is one distribution and $Q$ is the other. Let's see what this actually computes with a concrete example.

**A worked example.** Suppose you live in a city where it's sunny 70% of days and rainy 30% of days. That's the true distribution $P$. Your weather model predicts 50/50, sunny or rainy with equal probability. That's $Q$.

Let's compute $KL(P \| Q)$ step by step. There are only two outcomes, so the sum has two terms:

$$KL(P \| Q) = P(\text{sunny}) \log \frac{P(\text{sunny})}{Q(\text{sunny})} + P(\text{rainy}) \log \frac{P(\text{rainy})}{Q(\text{rainy})}$$

$$= 0.7 \log \frac{0.7}{0.5} + 0.3 \log \frac{0.3}{0.5}$$

$$= 0.7 \times 0.336 + 0.3 \times (-0.511) = 0.235 - 0.153 = 0.082 \text{ nats}$$

(Using natural log here. If we used log base 2, we'd get bits instead of nats, but the idea is the same.)

So $KL(P \| Q) \approx 0.082$. What does this number mean? It measures the "wasted information" from using $Q$ when the truth is $P$. Your 50/50 model over-allocates probability to rainy days (30% actual vs 50% predicted) and under-allocates to sunny days. That mismatch costs you 0.082 nats of efficiency per day.

Now here's something important. Let's compute the *reverse* direction: $KL(Q \| P)$. This swaps the roles of $P$ and $Q$:

$$KL(Q \| P) = Q(\text{sunny}) \log \frac{Q(\text{sunny})}{P(\text{sunny})} + Q(\text{rainy}) \log \frac{Q(\text{rainy})}{P(\text{rainy})}$$

$$= 0.5 \log \frac{0.5}{0.7} + 0.5 \log \frac{0.5}{0.3}$$

$$= 0.5 \times (-0.336) + 0.5 \times 0.511 = -0.168 + 0.256 = 0.088 \text{ nats}$$

Different number! $KL(P \| Q) \approx 0.082$ but $KL(Q \| P) \approx 0.088$. **KL divergence is not symmetric.** The "distance" from $P$ to $Q$ is not the same as from $Q$ to $P$. This might seem like a quirk, but it turns out to be one of the most consequential properties in all of ML. We'll see why shortly.

**Two intuitions for what KL measures.**

The first is the **coding cost** intuition. Imagine you need to send daily weather reports using a binary code. An efficient code assigns shorter codes to more likely events. If the true weather is $P$ (70/30) but you design your code for $Q$ (50/50), you'll waste bits. You're using equal-length codes for sunny and rainy, but sunny happens much more often and deserves a shorter code. KL divergence is exactly the average number of extra bits (or nats) wasted per message because of this mismatch.

The second is the **surprise** intuition. When an event happens, your "surprise" is $-\log Q(x)$. Rare events under your model are more surprising. KL divergence is the average *excess* surprise you experience when reality follows $P$ but you're expecting $Q$. In our weather example: on sunny days, your 50/50 model is *more* surprised than it should be, because it only expected 50% sun when the true rate is 70%. On rainy days, it's *less* surprised than it should be, because it expected 50% rain when rain only happens 30% of the time. The net excess surprise across many days (weighted by how often each actually happens) is the KL divergence, and it's always non-negative.

**Key properties.** KL divergence is always non-negative (you can never do *better* by using the wrong distribution). It's zero if and only if $P = Q$ exactly. And as we just saw, it's **not symmetric**: $KL(P \| Q) \neq KL(Q \| P)$. It's also not a true distance in the mathematical sense (it doesn't satisfy the triangle inequality). Despite that, it's the right measure for what MLE is doing, as we'll see next.

## MLE is secretly minimizing KL divergence

Post 2 stated this without proof and promised we'd walk through it here. The derivation is surprisingly short. Let's take it one step at a time.

**Step 1: Turn your data into a distribution.** You observed $n$ data points: $x_1, x_2, \ldots, x_n$. Imagine creating a histogram from these observations and treating it as a probability distribution. Call it $\hat{P}$. It puts a probability spike of $1/n$ on each observed point.

For example, if you flipped a coin 3 times and saw heads, heads, tails, then $\hat{P}(\text{heads}) = 2/3$ and $\hat{P}(\text{tails}) = 1/3$. Nothing fancy. It's just the relative frequencies in your data.

**Step 2: Write the KL divergence between this data distribution and your model.** Your model is $Q_\theta$, a distribution with adjustable parameters $\theta$. The KL divergence from the data to the model is:

$$KL(\hat{P} \| Q_\theta) = \sum_x \hat{P}(x) \log \frac{\hat{P}(x)}{Q_\theta(x)}$$

This measures how far your model is from the data histogram.

**Step 3: Split the log.** The log of a fraction is the difference of logs:

$$KL(\hat{P} \| Q_\theta) = \sum_x \hat{P}(x) \log \hat{P}(x) - \sum_x \hat{P}(x) \log Q_\theta(x)$$

Look at these two pieces:

- The first sum, $\sum_x \hat{P}(x) \log \hat{P}(x)$, is the negative entropy of the data, $-H(\hat{P})$. This depends only on the data you observed. It has **nothing to do with $\theta$**. When we minimize KL over $\theta$, this is just a constant we can ignore.

- The second sum, $-\sum_x \hat{P}(x) \log Q_\theta(x)$, is called the **cross-entropy** between the data and the model. This is the part that depends on $\theta$.

**Step 4: Replace the abstract sum with your actual data.** Since $\hat{P}$ puts mass $1/n$ on each observed data point, the cross-entropy simplifies to:

$$\sum_x \hat{P}(x) \log Q_\theta(x) = \frac{1}{n} \sum_{i=1}^{n} \log Q_\theta(x_i)$$

This is just the average log-likelihood over your data points. We've seen this before in Post 2.

**Step 5: Put it together.**

$$KL(\hat{P} \| Q_\theta) = \underbrace{-H(\hat{P})}_{\text{constant}} - \frac{1}{n} \sum_{i=1}^{n} \log Q_\theta(x_i)$$

Minimizing KL over $\theta$ means maximizing $\frac{1}{n} \sum_{i=1}^{n} \log Q_\theta(x_i)$, which is the average log-likelihood. That's exactly what MLE does. MLE and KL minimization are the same optimization, just viewed from different angles.

**One important detail: the direction.** Notice which distribution is which: the data $\hat{P}$ is in the first slot, the model $Q_\theta$ is in the second slot. This is $KL(\hat{P} \| Q_\theta)$, called **forward KL**. MLE minimizes forward KL. The direction matters enormously, because the two directions of KL produce very different behavior. That's the next section, and it's the most important one in this post.

## Forward vs reverse KL: the same formula, very different behavior

We saw that $KL(P \| Q) \neq KL(Q \| P)$. With the weather example, the numbers were close (0.082 vs 0.088), so you might think the asymmetry is a minor technicality. It's not. When you're fitting a model, the direction you minimize completely changes what the model learns.

Let's see why by going back to the formula and looking at it term by term.

**Forward KL: $KL(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$**

Each term in this sum is weighted by $P(x)$, the *true* distribution. So the sum only cares about regions where $P$ puts significant mass. Let's think about what this penalizes:

- If $P(x)$ is large but $Q(x)$ is small at some point $x$, the ratio $P(x)/Q(x)$ is huge, and the log of a huge number is a big penalty. In plain terms: if the truth says "this outcome is common" but your model says "this outcome is rare," you pay heavily.

- If $Q(x)$ is large but $P(x)$ is near zero, that term practically vanishes. The $P(x)$ weight out front kills it. Your model is wasting probability mass on something that doesn't happen, but forward KL doesn't care.

The upshot: forward KL forces $Q$ to **cover everywhere $P$ has mass**. It penalizes missing real outcomes but doesn't penalize hallucinating fake ones. This behavior is called "mass-covering" or "mean-seeking."

**Reverse KL: $KL(Q \| P) = \sum_x Q(x) \log \frac{Q(x)}{P(x)}$**

Now each term is weighted by $Q(x)$, the *model* distribution. The sum only cares about regions where $Q$ puts mass:

- If $Q(x)$ is large but $P(x)$ is small, the ratio $Q(x)/P(x)$ is huge, and that's a big penalty. Your model says "this is likely" but reality says "this almost never happens." Reverse KL punishes this harshly.

- If $P(x)$ is large but $Q(x)$ is near zero, no penalty. The $Q(x)$ weight out front kills the term. The truth says "this outcome is common" but your model ignores it entirely, and reverse KL is fine with that.

The upshot: reverse KL forces $Q$ to **only put mass where $P$ is strong**. It penalizes hallucinating but doesn't penalize missing real outcomes. This is called "mode-seeking" or "zero-forcing."

**A concrete example to see the difference.** Imagine the true distribution $P$ has two peaks (a mixture of two Gaussians, one centered at $-3$ and one at $+3$). Your model $Q$ is a single Gaussian, so it can only have one peak. Which peak does it choose?

*Minimizing forward KL* ($KL(P \| Q)$): $Q$ must cover both peaks. If $Q$ concentrates on just the left peak, it assigns near-zero probability to the right peak where $P$ has significant mass. That's a massive penalty because of the $P(x)$ weight. So $Q$ does the only thing it can: it stretches wide, centering near 0 with a large variance, trying to cover both peaks. The result is that $Q$ sits in the valley between the two modes, assigning a lot of probability to a region where $P$ actually has very little mass. It covers everything, but at the cost of hallucinating probability in the middle.

*Minimizing reverse KL* ($KL(Q \| P)$): $Q$ only gets penalized where *it* puts mass. If $Q$ concentrates entirely on the left peak, it assigns zero probability to the right peak. But that term has a $Q(x) \approx 0$ weight, so it contributes nothing to the sum. $Q$ locks perfectly onto one mode and completely ignores the other. The result is that $Q$ captures one peak well but misses the other entirely.

**The practical takeaway.** Forward KL says: "don't miss anything real, even if you have to hallucinate a little." Reverse KL says: "don't hallucinate anything, even if you miss something real." These are genuinely different philosophies, and different algorithms choose different ones:

- **MLE** (model training) minimizes forward KL, as we just proved. That's why trained models try to cover all the data, even at the cost of some probability in weird places.
- **RLHF** (model alignment) uses reverse KL in its penalty term, as we'll see next. That's why it prevents the model from generating text the base model would never produce, even if it means missing some valid outputs.

## KL divergence in RLHF

This is where KL divergence connects to the frontier of ML. If you've read about how ChatGPT or Claude gets fine-tuned from human feedback, the core idea is: take a pretrained language model, and adjust it to produce outputs that humans prefer. The tool that keeps this process from going off the rails is a KL penalty.

**The RLHF objective.** The goal is to find a policy $\pi$ (the fine-tuned model) that maximizes human reward while staying close to the pretrained base model $\pi_{\text{ref}}$:

$$\max_\pi \; \mathbb{E}_{x,y \sim \pi}[r(x, y)] - \beta \cdot KL(\pi \| \pi_{\text{ref}})$$

The first term says: generate responses that score high on the reward model (trained from human preferences using Bradley-Terry, as we saw in Post 1). The second term says: don't drift too far from the base model. The parameter $\beta$ controls the tradeoff.

**Why KL and not something simpler, like L2 distance on the weights?** Because KL operates on what the model *outputs* (probability distributions over text), not on the internal weights. Two models can have very different weights but produce similar outputs, or similar weights but very different outputs. KL measures the thing we actually care about: how differently the fine-tuned model behaves compared to the base model. Specifically, it penalizes the fine-tuned model for assigning high probability to tokens that the base model considers unlikely.

**The direction matters.** This is $KL(\pi \| \pi_{\text{ref}})$, which is reverse KL: the learned policy $\pi$ is in the first slot. Looking back at what we learned about reverse KL: it penalizes $\pi$ for putting mass where $\pi_{\text{ref}}$ doesn't. If the base model thinks some token sequence is near-impossible (say, random gibberish or a harmful output it was trained to avoid), and the fine-tuned model starts generating it to chase reward, the KL penalty pushes back hard. But if the base model assigns probability to some output that the fine-tuned model ignores (maybe a valid but low-reward response), reverse KL is fine with that.

This is exactly the right behavior for alignment. We want the fine-tuned model to be a "filtered" version of the base model, selecting the best outputs from what the base model can already produce, not inventing entirely new behaviors.

**The $\beta$ tradeoff.** This is one of the most important practical knobs in RLHF:

- $\beta$ too small: the model chases reward aggressively. It finds ways to exploit the reward model, producing outputs that score high but are actually nonsensical, repetitive, or degenerate. This is called **reward hacking**. The model learned to game the proxy (the reward model) instead of actually being helpful.

- $\beta$ too large: the model barely moves from the base. It stays safe but doesn't learn much from the human feedback. You've essentially paid for all the RLHF infrastructure and gotten minimal improvement.

Practitioners tune $\beta$ to find the sweet spot: enough freedom to improve, enough constraint to stay coherent.

**How KL is actually computed for language models.** A language model generates text one token at a time. The probability of a full response is:

$$\pi(y \mid x) = \pi(t_1 \mid x) \cdot \pi(t_2 \mid x, t_1) \cdot \pi(t_3 \mid x, t_1, t_2) \cdots$$

That's a product of per-token conditional probabilities. Now, KL divergence involves $\log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$. Taking the log of a product gives a sum:

$$\log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = \sum_{k=1}^{T} \log \frac{\pi(t_k \mid x, t_{<k})}{\pi_{\text{ref}}(t_k \mid x, t_{<k})}$$

So the KL between the two models decomposes into a sum of per-token terms. At each position, you compare the fine-tuned model's log-probability for that token with the base model's log-probability. Sum up the differences across all token positions, and you have the KL for that response. This is exactly what RLHF implementations compute in practice: run both models on the same text, compare their per-token log-probs, and sum.

Notice the same pattern from earlier: the sum comes from a product (here, the autoregressive factorization) turned into a sum by the log. The same structural trick behind Gaussian-to-MSE and independence-to-sum shows up here in language models.

## What's next

Let's take stock. We started with MSE and traced it back to the Gaussian assumption. We defined KL divergence and showed that MLE is secretly minimizing it. We saw that the two directions of KL (forward and reverse) produce fundamentally different behavior. And we ended at the RLHF objective, where a reverse-KL penalty keeps the fine-tuned model anchored to the base.

The RLHF objective has an interesting property: it turns out to have a closed-form solution. You can rearrange it algebraically to eliminate the reward model entirely, expressing the optimal policy directly in terms of preferences. That rearrangement is called **DPO** (Direct Preference Optimization), and it connects everything from the series so far: Bradley-Terry from Post 1, MLE from Post 2, and KL divergence from this post. That's the next post.

---

**References**

1. T. Cover and J. Thomas. *Elements of Information Theory*. The standard reference for entropy, cross-entropy, and KL divergence.
2. [RLHF Book, Chapter 8: Regularization.](https://rlhfbook.com/c/08-regularization.html) Covers the KL penalty in RLHF and why it's needed.
3. A. Kristiadi. ["KL Divergence: Forward vs Reverse?"](https://agustinus.kristia.de/blog/forward-reverse-kl/) Excellent visualizations of mode-seeking vs mass-covering behavior.
4. L. Mao. ["Cross Entropy, KL Divergence, and MLE."](https://leimao.github.io/blog/Cross-Entropy-KL-Divergence-MLE/) The equivalence between MLE and KL minimization, explained with derivations.
5. [Post 1 in this series]({% post_url 2026-02-08-logistic-regression-loss-from-bradley-terry %}). The Bradley-Terry / logistic loss derivation.
6. [Post 2 in this series]({% post_url 2026-02-09-what-is-maximum-likelihood-really %}). Maximum likelihood estimation and its properties.
