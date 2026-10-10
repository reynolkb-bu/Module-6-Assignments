# Week 7 Reflection

## 1. Create a linear regression model involving a confounder that is left out of the model. Show whether the true correlation between X and Y is overestimated, underestimated, or neither. Explain in words why this is the case for the given coefficients you have chosen.

In my model, W is the confounder. It causes both X and Y. The true effect of X on Y is 1.0, and W pushes Y up by 2.0. Then I fit the regression twice: once leaving W out, and once with W in. The full code is in `week7_reflection.py`.

```python
w = rng.normal(0, 1, n)
x = w + rng.normal(0, 1, n)
y = 1.0 * x + 2.0 * w + rng.normal(0, 1, n)
df = pd.DataFrame({"W": w, "X": x, "Y": y})

smf.ols("Y ~ X", df).fit().params["X"]       # 1.94  (W left out)
smf.ols("Y ~ X + W", df).fit().params["X"]   # 1.01  (W controlled)
```

With W left out, the effect of X is overestimated. It comes out at 1.94 when the real answer is 1.0, so it's almost double.

The reason is that W raises X and also raises Y. When W is left out, it ends up in the error term. So a row with a high X probably also has a high W, and that row's Y is high for two reasons: its X and its W. The regression can't tell those apart, so it gives X credit for W's part too. With these numbers, the extra credit works out to 2.0 (W's effect on Y) times 0.5 (how much X moves with W), which is +1.0. That is why we get about 2 instead of 1. Once W goes in the model, X goes back to 1.01.

The direction depends on the signs I picked. If I flip W's effect on Y to -2.0, so W raises X but lowers Y, the bias flips too. Then X comes out at -0.08, which is underestimated so badly that X looks like it does nothing or even hurts Y. It would only be "neither" if W had no effect on X or no effect on Y, because then it wouldn't be a confounder.

A real-world example is ice cream and sunburns. Hot weather makes people buy ice cream, and it also makes people get sunburned. If you leave the weather out, ice cream looks like it causes sunburns.

## 2. Perform a linear regression analysis in which one of the coefficients is zero, compute the p-value of that coefficient, run the analysis 1000 times and report the best (smallest) p-value. If the p-value is less than 0.05, does this mean the coefficient actually is nonzero? What is the problem with repeating the analysis?

W is in the regression, but it has no effect on Y at all. Its true coefficient is zero.

```python
def p_value_of_w(n=1_000):
    w = rng.normal(0, 1, n)
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"W": w, "X": x, "Y": y})
    return smf.ols("Y ~ X + W", df).fit().pvalues["W"]

p = np.array([p_value_of_w() for _ in range(1_000)])
p.min()   # 0.0022
```

Over 1,000 runs, the smallest p-value for W was **0.0022**. That is way under 0.05, and if it was the only run I showed someone, they would think W really matters. 42 of the 1,000 runs (4.2%) came in under 0.05, and the median was 0.50.

No, a p-value under 0.05 doesn't mean the coefficient is actually nonzero. A p-value of 0.05 means that if the real coefficient were zero, we would still see a result this extreme about 5% of the time just from noise. So when the true coefficient is zero, about 1 out of every 20 runs will still pass the test. That is right in line with the 42 out of 1,000 I got.

The problem with repeating the analysis is that every run is another chance to get a false positive. If you run it enough times, you are pretty much guaranteed to get a small p-value, and if you only report the best one, it looks like you found something real when it's just luck. This is like the coin-flip room from last week. If enough people flip, someone will get ten heads. This is called p-hacking, or the multiple comparisons problem. The p-value only means what it says if you run the test once and decide ahead of time what you are testing. If you do run a lot of tests, you have to make the cutoff stricter. For example, the Bonferroni correction divides 0.05 by the number of tests, so 0.05 / 1,000 = 0.00005. My best p-value of 0.0022 doesn't pass that.
