# Week 9 Reflection

## 1. Write some code that will use a simulation to estimate the standard deviation of the coefficient when there is heteroskedasticity. Compare these standard errors to those found via statsmodels OLS or a similar linear regression model.

Heteroskedasticity means some rows have more noise than others. A good real-world example is income and spending. People with low incomes all spend about the same, but people with high incomes are all over the place. Some spend everything and some save most of it.

In my mock data the true effect of X is 2, and the noise gets bigger as X gets farther from 0.

```python
x = rng.normal(0, 1, n)
y = 1 + 2 * x + rng.normal(0, 1, n) * 2 * np.abs(x)
```

I made 1,000 datasets, fit `Y ~ X` on each, and took the standard deviation of the coefficients.

| Method | Std. dev. of the coefficient |
|---|---|
| Simulation | 0.104 |
| statsmodels OLS | 0.065 |
| statsmodels robust (`HC1`) | 0.113 |

OLS is about 40% too small. It assumes every row has the same noise, but the noisiest rows are the ones that move the slope the most. The robust standard error lets each row have its own noise and comes much closer to the simulation.

## 2. Write some code that will use a simulation to estimate the standard deviation of the coefficient when errors are highly correlated / non-independent. Compare these standard errors to those found via statsmodels OLS or a similar linear regression model.

A good real-world example is daily temperature and ice cream sales. If today is hotter than usual, tomorrow probably is too. So 100 days of data is really just a few hot and cold stretches, not 100 separate days.

I built X and the error so each value is `rho` times the last one plus new noise. A higher `rho` means neighbouring rows are more alike.

```python
for i in range(1, n):
    err[i] = rho * err[i - 1] + np.sqrt(1 - rho**2) * rng.normal(0, 1)
```

1,000 datasets of 100 rows:

| rho | Simulation | OLS | Bootstrap |
|---|---|---|---|
| 0.00 | 0.105 | 0.097 | 0.094 |
| 0.50 | 0.129 | 0.095 | 0.091 |
| 0.90 | 0.286 | 0.131 | 0.148 |
| 0.99 | 0.575 | 0.115 | 0.100 |

With no correlation everything matches. As `rho` goes up, the real spread grows but OLS stays low. OLS thinks it has 100 independent rows when it really has a lot fewer.

## 3. Show that if the correlation between errors is high enough, then the estimated standard deviation of the coefficient, using bootstrap errors, might not match that found by a full simulation of the Data Generating Process.

The bootstrap misses too. At `rho = 0.99` it says 0.10 when the real answer is 0.57. The bootstrap resamples rows one at a time from a single dataset, so it breaks up the rows that were moving together. Every resample looks like independent data.

Think of shuffling a song. If you pick random single notes, you lose the melody. If you pick random chunks of a few seconds, the melody survives inside each chunk. The fix is the same: resample blocks of neighbouring rows instead of single rows. You need a huge amount of data for this to work, because each block has to be long and you still need a lot of blocks.

```python
starts = rng.integers(0, n - block + 1, n // block)
rows = (starts[:, None] + np.arange(block)).ravel()
```

At `rho = 0.99`:

| Rows | Simulation | Row bootstrap | Block bootstrap |
|---|---|---|---|
| 100 | 0.558 | 0.355 | 0.354 |
| 10,000 | 0.101 | 0.011 | 0.075 |
| 100,000 | 0.032 | 0.003 | 0.034 |

At 100,000 rows the block bootstrap is close to the simulation. More data alone doesn't fix the regular bootstrap though. It stays about 10x too small no matter how many rows you have.
