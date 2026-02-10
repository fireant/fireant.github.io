---
layout: post
title:  "What Is Maximum Likelihood, Really?"
date:   2026-02-09 12:00:00 -0700
categories: ML Principles
---

*This is the second post in a series called "ML Principles for Practitioners." The [first post]({% post_url 2026-02-08-logistic-regression-loss-from-bradley-terry %}) derived the logistic loss from the Bradley-Terry model and used maximum likelihood estimation without explaining it. This post fills that gap.*

---

Most ML practitioners have a rough sense of maximum likelihood estimation: find the parameters that make the data most probable. That's correct, but there's a lot more going on beneath the surface. This post covers the mechanics, then digs into what MLE really means, how it can fail, and why it has some surprising properties.

## The basic idea

You have data. You have a model with adjustable parameters. MLE finds the parameter values that make the observed data most probable under the model.

A simple example: you flip a coin 100 times and get 63 heads. If the model is "the coin has some bias $p$," then MLE sets $p = 0.63$, because that value makes the observed 63 heads most likely. Intuitive and hard to argue with.

## The mechanics

Let's walk through the coin example to see the concrete steps.

**Step 1: Write down the probability of the data given the parameters.** Each flip is independent with probability $p$ of heads. The probability of seeing 63 heads and 37 tails is:

$$L(p) = p^{63} (1-p)^{37}$$

This is the **likelihood function**. It's not a probability over $p$. It's the probability of the data, viewed as a function of $p$.

**Step 2: Take the log.** Products of many small numbers are numerically unstable and hard to differentiate. Taking the log turns them into sums:

$$\ell(p) = 63 \log p + 37 \log(1 - p)$$

This is the **log-likelihood**. Maximizing it gives the same answer as maximizing the likelihood, since log is monotonically increasing.

**Step 3: Find the maximum.** Take the derivative, set it to zero, solve:

$$\frac{d\ell}{dp} = \frac{63}{p} - \frac{37}{1-p} = 0 \quad \Rightarrow \quad p = \frac{63}{100} = 0.63$$

That's MLE. Write the probability of your data, take the log, maximize. For simple models you can solve analytically. For complex ones (neural networks, Bradley-Terry with many items) you maximize numerically with gradient ascent, or equivalently, minimize the negative log-likelihood with gradient descent.

**Step 4 (in practice): Negate the log-likelihood to get a loss function.** Optimizers minimize by convention, so we flip the sign. The negative log-likelihood becomes the loss:

$$\mathcal{L}(p) = -63 \log p - 37 \log(1 - p)$$

Minimizing this is the same as maximizing the likelihood.

In the [previous post]({% post_url 2026-02-08-logistic-regression-loss-from-bradley-terry %}), we did exactly this for the Bradley-Terry model: wrote the likelihood of the win/loss data, took the negative log, and got the binary cross-entropy loss. That was not a coincidence. It leads to the first insight.

## Insight 1: Choosing a loss function IS choosing a model

This is the most important practical takeaway in this post. Most practitioners think of the loss function as a training knob, something you pick from a menu. In reality, every standard loss function is the negative log-likelihood of a specific probabilistic model:

- **Mean squared error** = you're assuming the true value has Gaussian noise added to it
- **Binary cross-entropy** = you're assuming a Bernoulli/Bradley-Terry comparison model ([Post 1]({% post_url 2026-02-08-logistic-regression-loss-from-bradley-terry %}))
- **Categorical cross-entropy** = you're assuming a categorical distribution over classes

The pattern: pick a probabilistic model, write down the likelihood, take the negative log, and you get a loss function. Every loss function is the negative log-likelihood of *some* model, whether or not the person using it realizes it.

When you pick a loss, you're not just picking a way to measure error. You're making a claim about how the world generated your data. If that claim is wrong, the parameters MLE finds may not be what you want.

## Insight 2: MLE = minimizing KL divergence

There's an elegant information-theoretic way to see what MLE is doing. **KL divergence** is a measure of how different two probability distributions are (we'll explore it in depth in the next post). It turns out that minimizing the average negative log-likelihood is mathematically the same as minimizing the KL divergence between the data distribution and the model distribution. The proof is a short algebraic rearrangement, which we'll walk through next time.

The intuition: MLE finds the model that is least "surprised" by the data. Among all the distributions your model can represent, it picks the one closest to what actually happened. If you think of probability distributions as codebooks, MLE picks the codebook that wastes the fewest bits encoding the observations.

## Insight 3: MLE is only optimal in the long run

MLE has a strong theoretical guarantee: with enough data, no other consistent estimator can have lower variance. This limit is called the **Cramer-Rao lower bound**. The idea is that the data itself contains a finite amount of information about the parameter (formalized by the Fisher information), and that limits how precisely *any* estimator can pin down the true value. MLE, with enough data, hits that limit. It extracts all the information the data has to offer.

But that guarantee is *asymptotic*. It holds as data goes to infinity. With finite data, MLE can be biased.

The classic example: suppose your model is "the data points come from a Gaussian with unknown mean $\mu$ and unknown variance $\sigma^2$." Let's apply the mechanics from earlier. The likelihood for $n$ independent Gaussian observations is:

$$L(\mu, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

Take the log:

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

Maximize with respect to $\mu$ and $\sigma^2$ (take derivatives, set to zero). The MLE for the mean turns out to be the sample average $\hat{\mu} = \bar{x}$, no surprise. The MLE for the variance is:

$$\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2$$

Notice this also shows Insight 1 in action. If you negate the log-likelihood above and drop the constant terms, what's left to minimize is $\sum(x_i - \mu)^2$, which is the mean squared error. MSE is the negative log-likelihood of a Gaussian model.

Now here's the problem. The MLE divides by $n$, but the unbiased estimate divides by $n - 1$. The MLE systematically underestimates the variance. Why? Because it measures how far each point is from the sample mean $\bar{x}$, but treats $\bar{x}$ as if it were the true mean. Since $\bar{x}$ is fit to the same data, it sits artificially close to the points, making the residuals look smaller than they really are. The $n-1$ correction accounts for this. With 5 data points, the difference is 20%. With 1000 data points, it's negligible.

The practical takeaway: with small datasets, MLE overfits. It fits the noise in the data, not just the signal. This is not a bug in your training loop. It's a fundamental property of MLE. You're asking "what parameters make *this specific data* most likely?" and with little data, the answer is too tailored to the specific sample you happened to see.

This is why regularization exists, and we'll cover it in a future post.

## Insight 4: MLE can completely fail to exist

Sometimes MLE doesn't just give a biased answer. It gives no answer at all.

To see why, think about what the sigmoid does. Recall from [Post 1]({% post_url 2026-02-08-logistic-regression-loss-from-bradley-terry %}): the predicted probability is $\sigma(w \cdot x)$, and the loss for a correctly classified point is $-\log \sigma(w \cdot x)$. The sigmoid never actually reaches 1. It only approaches it as its input grows. So $-\log \sigma(w \cdot x)$ never reaches 0. It only gets closer to 0 as $w \cdot x$ gets larger.

Now imagine the data is **perfectly separable**: there's some direction in feature space where all the positive examples are on one side and all the negative examples are on the other. The model can classify every training point correctly. The question is: how confident should it be?

MLE says: as confident as possible. Making $w$ larger increases $w \cdot x$ for correctly classified points, which pushes $\sigma(w \cdot x)$ closer to 1, which reduces the loss. And there's no counterforce, because no points are on the wrong side. So the optimal $w$ is infinitely large. The MLE doesn't exist as a finite number.

In practice, this means the optimizer never converges. Each step improves the loss a little by making the weights a little bigger. The loss keeps decreasing but never reaches a minimum. If you plot the weights over training steps, they drift upward without settling.

This happens more often than you might think: high-dimensional data with few samples, or datasets where one class is rare and happens to be cleanly separable in some feature direction.

The Bradley-Terry version is even more intuitive. If one team beat every other team in every single matchup, what should its strength score be? Under MLE, infinity. Any finite score $s$ gives a probability of winning that's less than 1, but this team won every game, so a higher $s$ always improves the likelihood. There's no finite optimum.

The fix: regularization (adding a penalty that keeps weights finite) or Bayesian methods (putting a prior on the parameters). Both add a counterforce that resists weights growing without bound, so the optimization has a finite solution.

## Insight 5: MLE with a wrong model still does something sensible

Every model is wrong. The Gaussian assumption behind MSE is never exactly true. The Bradley-Terry assumption behind cross-entropy is an approximation. So what does MLE do when the model is wrong?

Something useful, it turns out. White's theorem (1982) says: even when the model is misspecified, MLE converges to the parameter values that make your model as close as possible to the true data distribution, measured in KL divergence.

In plain terms: MLE finds the best approximation within your model family. If the true relationship is a wiggly curve and your model is a straight line, MLE gives you the straight line that is closest to the wiggly curve (in a KL sense). It degrades gracefully. The predictions are still the best your model family can do.

But there's a catch. When the model is wrong, the standard formulas for confidence intervals and standard errors become incorrect. They assume the model is right, so they underestimate uncertainty. In practice, if you're reporting confidence intervals from a model that might be wrong (and it always is), use **robust standard errors** instead of the default ones. Most statistical software supports this. The key point for practitioners: MLE predictions degrade gracefully, but MLE uncertainty estimates do not.

## Insight 6: The invariance property

MLE has a useful property that not all estimators share: if $\hat{\theta}$ is the MLE of $\theta$, then $g(\hat{\theta})$ is the MLE of $g(\theta)$ for any function $g$.

The reason this works is because of how MLE is defined. MLE is an **argmax**: it finds the parameter value where the likelihood function reaches its peak. When you transform the parameter through some function $g$, the peak of the likelihood doesn't move. It stays at the same point, just with a new label. An argmax is a structural property of the function that survives relabeling.

Compare this to how other estimators are defined. The unbiased variance estimator from Insight 3 (the one that divides by $n-1$) is defined by the property that its **expected value** equals the true variance: $E[\hat{\sigma}^2] = \sigma^2$. Now take the square root to get a standard deviation estimate. Is $E[\sqrt{\hat{\sigma}^2}] = \sigma$? No. In general, $E[g(X)] \neq g(E[X])$ for nonlinear $g$. This is Jensen's inequality, and it's why unbiasedness breaks under transformation. The property that defines the estimator involves an expectation, and expectations don't commute with nonlinear functions.

MLE sidesteps this entirely. It's not defined through an expectation or a moment equation. It's defined as "the peak." Peaks just get relabeled under transformation. That's the fundamental reason MLE has invariance and most other estimators don't.

Why does this matter in practice? The same quantity often shows up in different forms. A logistic regression gives you log-odds, but you might want probabilities. A Poisson model gives you a rate $\lambda$, but you might want the mean waiting time $1/\lambda$. The invariance property says you don't need to re-derive or re-fit anything. Just transform your MLE estimate and you have the MLE of the transformed quantity. No correction needed.

For example, if your MLE for the log-odds of an event is 1.5, then the MLE for the odds is $e^{1.5} \approx 4.48$, and the MLE for the probability is $\sigma(1.5) \approx 0.82$. You get all three for free from one fit.

In the context of Post 1, this is why the Bradley-Terry model works the same whether you parameterize strengths as $s_i$ (log-scale) or as $e^{s_i}$ (original scale). The MLE is consistent either way.

## What MLE doesn't give you

MLE gives point estimates, not uncertainty. It tells you the single best parameter value, but not how confident you should be. For that, you need confidence intervals (frequentist) or posterior distributions (Bayesian).

MLE assumes you have the right model family. If you don't, the predictions are still useful (Insight 5), but the uncertainty quantification breaks.

MLE with too little data overfits (Insight 3) and can even fail to exist (Insight 4). The principled fix is to add prior information about the parameters, turning MLE into MAP (maximum a posteriori) estimation. That's the idea behind regularization.

This post introduced the idea that MLE minimizes KL divergence (Insight 2). The next post will go much deeper into KL divergence itself, because it turns out to be the single most important concept for understanding modern alignment techniques like RLHF and DPO.

---

**References**

1. H. White. "Maximum Likelihood Estimation of Misspecified Models." *Econometrica*, 1982. The foundational paper showing MLE converges to the KL-closest model even under misspecification.
2. [MLE under a misspecified model](https://andrewcharlesjones.github.io/journal/mlemisspecified.html). Accessible blog treatment of White's theorem.
3. [Cross Entropy, KL Divergence, and MLE](https://leimao.github.io/blog/Cross-Entropy-KL-Divergence-MLE/). The equivalence between MLE and KL minimization, explained.
4. [Post 1 in this series]({% post_url 2026-02-08-logistic-regression-loss-from-bradley-terry %}). The Bradley-Terry / logistic loss derivation that motivates this post.
