# Week 2 Reflection

## 1. Invent an example situation that would use fixed effects.

A coffee chain with 30 stores rolls out a new menu and wants to know whether it raised weekly sales.

The problem is the stores were never comparable. The airport store does huge volume because of foot traffic; the suburban strip-mall store does a fraction of that. Compare sales across stores and you are mostly measuring location, not the menu.

**Fixed effects give each store its own baseline.** Every store gets its own intercept, and one shared menu effect is estimated on top of that. The question stops being "did the airport store outsell the suburban store" and becomes "did each store beat its own normal week." Anything about a store that stays constant, including things never measured like parking or the manager, gets absorbed into that intercept and stops contaminating the answer.

That is the same model as `week2.py`: one slope on time shared by all groups, plus a separate intercept per group. The cost is that only things that change can be studied. A store that never switched menus adds nothing, and anything permanently true about a store is now invisible.

## 2. Bootstrap the variance in the mean of a Pareto distribution. Explain what you had to do. As the sample size grows, what happens to that variance?

Program: `week2_pareto_bootstrap.py`. **Answer: the variance gets smaller, at roughly 1/n.**

**Variance of what.** Two variances are easy to conflate here. **var(X)** is the spread of the individual values — 0.75 for the Pareto I used, and n never changes it. **var(mean)** is how much the sample average itself moves from one sample to the next. That is what shrinks. The poll makes it obvious: individual answers stay all over the place no matter how many people you survey, but the poll's *average* barely moves when you re-run it, and moves less the more people you ask.

**What I did.** You only ever get one sample in real work, so the bootstrap fakes the re-running: treat your sample as a stand-in for the population and redraw from it with replacement. I drew a sample of size n from a Pareto, resampled it with replacement and took the mean 5,000 times, took the variance of those 5,000 means, and repeated for n = 100 → 25,600, quadrupling each time so the rate is easy to read.

**Two things needed care.** numpy's `pareto()` starts at 0, so I shifted it by 1. More importantly, my first version used one sample per n and the ratios came out 0.140, 0.667, 0.173, 0.197 — no pattern at all. Pareto is heavy-tailed, so whether a single sample catches one huge value swings the estimate; averaging over 40 independent samples per n made the trend appear.

| n | bootstrap var(mean) | theory var(X)/n | ratio vs previous | where the average lands |
|---|---|---|---|---|
| 100 | 0.014596 | 0.007500 | - | 1.33 to 1.67 |
| 400 | 0.001947 | 0.001875 | 0.133 | 1.41 to 1.59 |
| 1,600 | 0.000412 | 0.000469 | 0.211 | 1.46 to 1.54 |
| 6,400 | 0.000105 | 0.000117 | 0.255 | 1.48 to 1.52 |
| 25,600 | 0.000030 | 0.000029 | 0.281 | 1.489 to 1.511 |

The last column is the true mean of 1.5 give or take two standard deviations. Sample size quadruples each row, so 1/n predicts each variance is about 0.25 of the one above it. The ratios settle right around that, and by n = 25,600 the bootstrap matches theory to the last digit — while var(X) is still 0.75 in every row.

**Caveat: the variance only exists for a Pareto shape above 2.** I used 3.0. At 1.5 the same program gives ratios of 0.035, 0.064, 0.251, 0.896 — no stable rate, because that tail has infinite variance and nothing to converge to. Income and insurance claims behave this way, where one catastrophic value outweighs the rest of the sample, and a bigger sample does not reliably buy a steadier average.
