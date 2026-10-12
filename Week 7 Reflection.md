# Week 7 Reflection

## 1. Create a linear regression model involving a confounder that is left out of the model. Show whether the true correlation between X and Y is overestimated, underestimated, or neither. Explain in words why this is the case for the given coefficients you have chosen.

The true correlation between X and Y is overestimated. My example is tutoring and test scores:

- **X** = hours of tutoring
- **Y** = test score
- **W** = how motivated the student is (the confounder)

Motivation affects both because students that are motivated sign up for more tutoring and they also score higher on their own. In my mock data, every hour of tutoring adds one point to the test score and motivation adds two points. 

```python
w = rng.normal(0, 1, N)                       # motivation
x = w + rng.normal(0, 1, N)                   # tutoring
y = 1.0 * x + 2.0 * w + rng.normal(0, 1, N)   # test score
```

| Model | Effect of tutoring |
|---|---|
| True effect | 1.00 |
| Leave motivation out (`Y ~ X`) | 1.94 |
| Include motivation (`Y ~ X + W`) | 1.01 |

If we take motivation out, then tutoring looks almost twice as good as it really is. This is because the model sees that students with tutoring scores higher, but it doesn't factor in that these students were also motivated. This means that tutoring would also take the credit of motivation. If motivation instead lowered scores, tutoring would come out at -0.08, which is underestimated, and tutoring would look like it had no effect.

## 2. Perform a linear regression analysis in which one of the coefficients is zero, compute the p-value of that coefficient, run the analysis 1000 times and report the best (smallest) p-value. If the p-value is less than 0.05, does this mean the coefficient actually is nonzero? What is the problem with repeating the analysis?

In my example, W has no effect on Y.

```python
w = rng.normal(0, 1, N)
x = rng.normal(0, 1, N)
y = 2 * x + rng.normal(0, 1, N)     # W is not in here
smf.ols("Y ~ X + W", df).fit().pvalues["W"]
```

I ran this analysis 1,000 times. The smallest p-value from these runs was 0.0022. 42 of the 1,000 runs were under 0.05.

No, a p-value under 0.05 does not mean the coefficient is actually nonzero. W has no effect, but in about 1 in 20 runs it passes by luck.

The problem is the more you repeat the analysis, each run has a chance to get a false positive. This means that if you run it enough times and only report the smallest fee value, there's a chance it could be the 1 in 20 runs that passes by luck.
