# Week 9 Reflection

## 1. Write some code that will use a simulation to estimate the standard deviation of the coefficient when there is heteroskedasticity. Compare these standard errors to those found via statsmodels OLS or a similar linear regression model.

Heteroskedasticity means the noise isn't the same size for every row. A good real-world example is income and spending. People with low incomes all spend about the same, but people with high incomes spread out a lot more. Some spend most of it and some save most of it. In my mock data the true effect of X is 2, and the noise gets bigger the farther X is from 0.

```python
x = rng.normal(0, 1, n)
y = 1 + 2 * x + rng.normal(0, 1, n) * 2 * np.abs(x)   # error sd = 2|X|
```

I made 1,000 datasets of 1,000 rows, fit `Y ~ X` on each one, and took the standard deviation of the 1,000 coefficients. That is the real spread of the estimate. Then I compared it to the standard error that statsmodels reports.

| Method | Std. dev. of the coefficient |
|---|---|
| Simulation (1,000 datasets) | 0.104 |
| statsmodels OLS standard error | 0.063 |
| statsmodels robust standard error (`HC1`) | 0.104 |

The OLS standard error is too small by about 40%. OLS assumes every row has the same amount of noise, so it averages the noise over all the rows. But the rows with the most noise are the rows where X is far from 0, and those are the rows that move the slope the most. OLS doesn't factor that in. Setting `cov_type="HC1"` lets each row have its own noise, and it matches the simulation.

## 2. Write some code that will use a simulation to estimate the standard deviation of the coefficient when errors are highly correlated / non-independent. Compare these standard errors to those found via statsmodels OLS or a similar linear regression model.

For correlated errors I used an AR(1) series, which is like the `make_error` function from the week 7 quiz. Each error is `rho` times the last one plus some new noise. I scaled the new noise so every error has a variance of 1, which means only the correlation changes when I change `rho`. Both X and the error are built this way.

```python
def make_error(rho, n):
    err = np.empty(n)
    err[0] = rng.normal(0, 1)
    for i in range(1, n):
        err[i] = rho * err[i - 1] + np.sqrt(1 - rho**2) * rng.normal(0, 1)
    return err

x = make_error(rho, n)
y = 1 + 2 * x + make_error(rho, n)
```

A real-world example is daily temperature and ice cream sales. If today is hotter than usual, tomorrow probably is too. So 100 days of data is really more like a handful of hot and cold stretches, not 100 separate days.

1,000 datasets of 100 rows each:

| rho | Simulated std. dev. | OLS standard error | Bootstrap (one dataset) |
|---|---|---|---|
| 0.00 | 0.105 | 0.096 | 0.099 |
| 0.50 | 0.129 | 0.085 | 0.076 |
| 0.90 | 0.286 | 0.104 | 0.093 |
| 0.99 | 0.554 | 0.265 | 0.209 |

When `rho` is 0 everything matches. As `rho` goes up, the real spread grows but the OLS standard error stays pretty low. OLS thinks it has 100 independent rows when it really has a lot less.

## 3. Show that if the correlation between errors is high enough, then the estimated standard deviation of the coefficient, using bootstrap errors, might not match that found by a full simulation of the Data Generating Process.

The usual bootstrap takes one dataset and resamples its rows with replacement over and over. In the table above, it misses just like OLS did. At `rho = 0.99` the bootstrap says 0.21 when the real answer is 0.55.

The bootstrap only has one dataset to work with. When it picks rows one at a time, it breaks up the neighbours that were moving together, so every resample looks like independent data. It also can't see the stretches that never showed up in that one dataset.

The fix is to resample blocks of neighbouring rows instead of single rows. That keeps the correlation inside each block. The catch is you need a huge amount of data. Each block has to be much longer than the correlation lasts (about 100 rows at `rho = 0.99`), and you still need a lot of blocks.

```python
starts = rng.integers(0, n - block + 1, n // block)
rows = (starts[:, None] + np.arange(block)).ravel()
coefs.append(slope(x[rows], y[rows]))
```

At `rho = 0.99`, 200 full simulations vs. one bootstrap dataset:

| Rows | Simulated std. dev. | Row bootstrap | Block bootstrap |
|---|---|---|---|
| 100 | 0.543 | 0.075 | 0.071 (blocks of 1) |
| 10,000 | 0.102 | 0.009 | 0.075 (blocks of 100) |
| 100,000 | 0.032 | 0.003 | 0.031 (blocks of 1,000) |

With 100,000 rows the block bootstrap finally matches the full simulation. A good thing to point out is that just adding more data doesn't fix the normal row bootstrap. It stays about 10x too small no matter how many rows there are. The huge amount of data only helps if you also resample in blocks.
