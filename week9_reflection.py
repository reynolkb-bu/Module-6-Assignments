"""Week 9 reflection -- simulated sd of the coefficient vs. what OLS reports.

Q1  Heteroskedasticity: the noise in Y grows with |X|.
Q2  Correlated errors: X and the error each follow an AR(1) series, then a
    bootstrap from one dataset vs. a full simulation of the DGP.

In every case the true model is Y = 1 + 2X + error, so the "real" sd of the
coefficient is the sd of the estimates across many fresh datasets.
"""

import numpy as np
import statsmodels.api as sm

N_ROWS = 1_000
N_EXPERIMENTS = 1_000
N_BOOT = 500
RHOS = [0.0, 0.5, 0.9, 0.99]
RHO_HIGH = 0.99
BIG_NS = [100, 10_000, 100_000]
BIG_EXPERIMENTS = 200
BIG_BOOT = 300
SEED = 0
rng = np.random.default_rng(SEED)


def ols(x, y, cov_type="nonrobust"):
    """(coef of X, its standard error) from Y ~ X."""
    fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type=cov_type)
    return fit.params[1], fit.bse[1]


# --- Q1: heteroskedasticity -------------------------------------------------

def hetero_data(n=N_ROWS):
    x = rng.normal(0, 1, n)
    y = 1 + 2 * x + rng.normal(0, 1, n) * 2 * np.abs(x)   # error sd = 2|X|
    return x, y


def report_hetero():
    runs = np.array([ols(*hetero_data()) for _ in range(N_EXPERIMENTS)])
    x, y = hetero_data()
    print(f"Q1 heteroskedastic errors, {N_EXPERIMENTS} datasets of {N_ROWS} rows:")
    print(f"  sd of coef across datasets:  {runs[:, 0].std(ddof=1):.4f}")
    print(f"  mean OLS standard error:     {runs[:, 1].mean():.4f}")
    # HC1 lets each row have its own error variance instead of one shared one.
    print(f"  robust (HC1) SE, one dataset: {ols(x, y, 'HC1')[1]:.4f}")


# --- Q2: correlated errors --------------------------------------------------

def make_error(rho, n):
    """AR(1) series. Each value is rho times the last plus new noise, scaled so
    every value has variance 1 -- only the correlation changes with rho."""
    err = np.empty(n)
    err[0] = rng.normal(0, 1)
    for i in range(1, n):
        err[i] = rho * err[i - 1] + np.sqrt(1 - rho**2) * rng.normal(0, 1)
    return err


def correlated_data(rho, n):
    x = make_error(rho, n)
    y = 1 + 2 * x + make_error(rho, n)
    return x, y


def slope(x, y):
    return np.polyfit(x, y, 1)[0]


def bootstrap_sd(x, y, block=1, n_boot=N_BOOT):
    """sd of the coef over resamples of one dataset.

    block = 1 is the usual bootstrap (resample single rows). A larger block
    resamples runs of `block` neighbouring rows, which keeps the correlation
    between neighbours inside each run.
    """
    n = len(x)
    coefs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, n // block)
        rows = (starts[:, None] + np.arange(block)).ravel()
        coefs.append(slope(x[rows], y[rows]))
    return np.std(coefs, ddof=1)


def report_correlated():
    n = 100
    print(f"\nQ2 correlated errors, {N_EXPERIMENTS} datasets of {n} rows:")
    print(f"{'rho':>5} {'simulated sd':>13} {'OLS SE':>8} {'bootstrap':>10}")
    for rho in RHOS:
        sims = [slope(*correlated_data(rho, n)) for _ in range(N_EXPERIMENTS)]
        x, y = correlated_data(rho, n)
        print(f"{rho:5.2f} {np.std(sims, ddof=1):13.4f} {ols(x, y)[1]:8.4f} "
              f"{bootstrap_sd(x, y):10.4f}")


def report_big_data():
    print(f"\n   rho = {RHO_HIGH}, bootstrap from one dataset vs. {BIG_EXPERIMENTS} full simulations:")
    print(f"{'rows':>8} {'simulated sd':>13} {'row bootstrap':>14} {'block bootstrap':>16}")
    for n in BIG_NS:
        sims = [slope(*correlated_data(RHO_HIGH, n)) for _ in range(BIG_EXPERIMENTS)]
        x, y = correlated_data(RHO_HIGH, n)
        block = max(n // 100, 1)   # 100 blocks, each 1/100th of the data
        print(f"{n:8,d} {np.std(sims, ddof=1):13.4f} {bootstrap_sd(x, y, 1, BIG_BOOT):14.4f} "
              f"{bootstrap_sd(x, y, block, BIG_BOOT):16.4f}  (block = {block:,d})")


def report():
    report_hetero()
    report_correlated()
    report_big_data()


if __name__ == "__main__":
    report()
