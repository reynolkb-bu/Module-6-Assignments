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

OLS is about 40% too small. OLS makes the assumption that each row contains the same amount of noise. However, the noisiest rows are the ones that move the slope the most. By using a standard error, it allows each row to have its own noise which comes a lot closer to the simulation.

## 2. Write some code that will use a simulation to estimate the standard deviation of the coefficient when errors are highly correlated / non-independent. Compare these standard errors to those found via statsmodels OLS or a similar linear regression model.

A good example to estimate the standard deviation of the coefficient when errors are highly correlated / non-independent is daily temperature and ice cream sales. You can assume that if today is hotter than normal, tomorrow probably will be too. Therefore, 100 days of data is essentially a couple of hot stretches and cold stretches.

I built X and the error with each value being `rho` times the last one plus new noise. If you have a higher `rho` it means that neighboring rows are similar.

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

If there is no correlation it means that everything matches. As `rho` goes up, the spread increases and OLS stays low. Because of this, OLS thinks it has 100 independent rows when in reality it has a lot fewer.

## 3. Show that if the correlation between errors is high enough, then the estimated standard deviation of the coefficient, using bootstrap errors, might not match that found by a full simulation of the Data Generating Process.

If the correlation between errors is high enough, then the estimated standard deviation of the coefficient might not match what is found by a full simulation. At rho = 0.99, the bootstrap stays at 0.10 when the real answer is 0.57. Bootstrap resamples each row one at a time from a single dataset. This means it breaks up the rows that were moving together so that each resample looks like independent data.

For example, you can think of shuffling a single song. If you pick random notes, you don’t get the overall melody. If you pick random chunks, then you do get the melody. The fix here is the same, we resample chunks of neighboring rows instead of single rows. You do need a lot of data for this to work since each chunk has to be long enough and you need a lot of chunks to analyze.

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

At 100,000 rows we are able to get the bootstrap close to the simulation. If you only add more data, it doesn’t fix the regular bootstrap. It stays 10x too small no matter how many rows you have.