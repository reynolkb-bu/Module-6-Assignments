# Week 2 Reflection

## 1. Invent an example situation that would use fixed effects.

A coffee chain with 30 stores rolls out a new menu and wants to know whether it raised weekly sales. The problem is that the stores were never comparable. An airport store sells far more than a suburban strip-mall store, so comparing stores mostly measures location, not the menu.

Fixed effects fix this by giving each store its own baseline. Every store gets its own intercept, and one shared menu effect is estimated on top. The question becomes "did each store beat its own normal week?" Anything about a store that doesn't change, like parking or foot traffic, is absorbed by its intercept.

This is the same model as `week2.py`: one shared slope plus a separate intercept per group. The tradeoff is that you can only study things that change. Anything permanently true about a store can't be measured.

## 2. Bootstrap the variance in the mean of a Pareto distribution. Explain what you had to do. As the sample size grows, what happens to that variance?

The variance of the mean gets smaller, at about 1/n (program: `week2_pareto_bootstrap.py`). This is different from the variance of the individual values, which stays at 0.75 no matter the sample size. Individual values stay spread out, but the average settles down as the sample grows.

To bootstrap it, I drew a Pareto sample of size n, resampled it with replacement 5,000 times, took the mean each time, and took the variance of those means. I repeated this for n from 100 to 25,600, quadrupling each time. Pareto has a heavy tail, so a single sample gave jumpy results. Averaging over 40 samples per n made the pattern clear.

Each 4x increase in n cut the variance to roughly a quarter, and by n = 25,600 the bootstrap (0.000030) matched theory (0.000029). One caveat: this only works when the Pareto shape is above 2. I used 3. With a heavier tail the variance is infinite, and a bigger sample doesn't reliably give a steadier average.
