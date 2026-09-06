# Week 2 Reflection

## 1. Invent an example situation that would use fixed effects.

A coffee chain with 30 stores rolls out a new menu and wants to know whether it raised weekly sales.

The problem is the stores were never comparable. The airport store does huge volume because of foot traffic; the suburban strip-mall store does a fraction of that. Compare sales across stores and you are mostly measuring location, not the menu.

**Fixed effects give each store its own baseline.** Every store gets its own intercept, and one shared menu effect is estimated on top of that. The question stops being "did the airport store outsell the suburban store" and becomes "did each store beat its own normal week." Anything about a store that stays constant, including things never measured like parking or the manager, gets absorbed into that intercept and stops contaminating the answer.

That is the same model as `week2.py`: one slope on time shared by all groups, plus a separate intercept per group. The cost is that only things that change can be studied. A store that never switched menus adds nothing, and anything permanently true about a store is now invisible.

## 2. Bootstrap the variance in the mean of a Pareto distribution. Explain what you had to do. As the sample size grows, what happens to that variance?

Program: `week2_pareto_bootstrap.py`.

**Answer: the variance gets smaller, at roughly 1/n.**

**Variance of what, exactly.** There are two different variances here and it is easy to conflate them.

The first is **var(X), the spread of the individual values**. For the Pareto I used, that is 0.75, and it is a fixed property of the distribution. Drawing more values does not change it. A bigger sample does not make individual draws cluster together.

The second is **var(mean), how much the sample average itself moves from one sample to the next**. That is what the table below reports, and that is what shrinks.

The poll makes the difference obvious. Individual people's answers are all over the place, and surveying more people does not change that. But the *poll's average* barely moves when you re-run it, and it moves less the more people you ask. It is the average that steadies, not the responses.

Concretely, with a true mean of 1.5:

| n | var(mean) | where the sample average usually lands |
|---|---|---|
| 100 | 0.007500 | 1.33 to 1.67 |
| 1,600 | 0.000469 | 1.46 to 1.54 |
| 25,600 | 0.000029 | 1.489 to 1.511 |

var(X) is 0.75 in every one of those rows. Only the average tightens up.

**Why bootstrap.** That re-running is exactly what you cannot do in real work: the pollster gets one sample, not many. The bootstrap fakes it by treating the sample you have as a stand-in for the population and redrawing from it with replacement. The spread of those redrawn averages estimates how much your one real average could have moved.

**What I did.**

1. Draw one sample of size n from a Pareto distribution.
2. Resample n rows from it with replacement, take the mean, repeat 5,000 times.
3. Take the variance of those 5,000 means.
4. Repeat for n = 100 → 25,600, quadrupling each time so the rate is easy to read.

**Two things needed care.** numpy's `pareto()` starts at 0, so I had to shift it by 1 to get the standard Pareto. More importantly, my first version used one sample per n and the ratios came out 0.140, 0.667, 0.173, 0.197 — no pattern at all. Pareto is heavy-tailed, so whether a single sample catches one huge value swings the estimate. I had to average over 40 independent samples per n before the trend appeared.

| n | bootstrap var(mean) | theory var(X)/n | ratio vs previous |
|---|---|---|---|
| 100 | 0.014596 | 0.007500 | - |
| 400 | 0.001947 | 0.001875 | 0.133 |
| 1,600 | 0.000412 | 0.000469 | 0.211 |
| 6,400 | 0.000105 | 0.000117 | 0.255 |
| 25,600 | 0.000030 | 0.000029 | 0.281 |

Sample size quadruples each row, so 1/n predicts each variance is about 0.25 of the one above it. The measured ratios settle right around that, and by n = 25,600 the bootstrap estimate matches theory to the last digit.

**Caveat: this needs a Pareto shape above 2, so the variance exists at all.** I used 3.0. At shape 1.5 the same program gives ratios of 0.035, 0.064, 0.251, 0.896 — no stable rate, because that tail has infinite variance and there is nothing to converge to. Insurance claims and income behave this way, where one catastrophic value outweighs the rest of the sample. With data like that, a bigger sample does not reliably buy a steadier average.
