"""Week 9 reflection -- simulated sd of the coefficient vs. what OLS and the
bootstrap report. The true model is always Y = 1 + 2X + error.

Q1  Heteroskedasticity: the noise in Y grows with |X|.
Q2  Correlated errors: X and the error are both AR(1) series.
"""

import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(0)


def ols_se(x, y, cov_type="nonrobust"):
    return sm.OLS(y, sm.add_constant(x)).fit(cov_type=cov_type).bse[1]


def slope(x, y):
    return np.polyfit(x, y, 1)[0]


def sim_sd(make_data, n_datasets):
    """sd of the coef across many fresh datasets -- the real answer."""
    return np.std([slope(*make_data()) for _ in range(n_datasets)], ddof=1)


# --- Q1: heteroskedasticity -------------------------------------------------

def hetero_data(n=1_000):
    x = rng.normal(0, 1, n)
    y = 1 + 2 * x + rng.normal(0, 1, n) * 2 * np.abs(x)   # error sd = 2|X|
    return x, y


# --- Q2: correlated errors --------------------------------------------------

def make_error(rho, n):
    """Each value is rho times the last plus new noise, scaled to variance 1."""
    err = np.empty(n)
    err[0] = rng.normal(0, 1)
    for i in range(1, n):
        err[i] = rho * err[i - 1] + np.sqrt(1 - rho**2) * rng.normal(0, 1)
    return err


def correlated_data(rho, n):
    x = make_error(rho, n)
    return x, 1 + 2 * x + make_error(rho, n)


def bootstrap_sd(x, y, block=1, n_boot=300):
    """Resample runs of `block` neighbouring rows (block = 1 is the usual bootstrap)."""
    n = len(x)
    coefs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, n // block)
        rows = (starts[:, None] + np.arange(block)).ravel()
        coefs.append(slope(x[rows], y[rows]))
    return np.std(coefs, ddof=1)


def report():
    x, y = hetero_data()
    print("Q1 heteroskedastic errors:")
    print(f"  simulated sd {sim_sd(hetero_data, 1_000):.4f}   OLS SE {ols_se(x, y):.4f}   "
          f"robust SE {ols_se(x, y, 'HC1'):.4f}")

    print("\nQ2 correlated errors, 100 rows:")
    print(f"{'rho':>5} {'simulated':>10} {'OLS':>7} {'bootstrap':>10}")
    for rho in [0.0, 0.5, 0.9, 0.99]:
        x, y = correlated_data(rho, 100)
        sd = sim_sd(lambda: correlated_data(rho, 100), 1_000)
        print(f"{rho:5.2f} {sd:10.4f} {ols_se(x, y):7.4f} {bootstrap_sd(x, y):10.4f}")

    print("\n   rho = 0.99, more data:")
    print(f"{'rows':>8} {'simulated':>10} {'row boot':>9} {'block boot':>11}")
    for n in [100, 10_000, 100_000]:
        x, y = correlated_data(0.99, n)
        sd = sim_sd(lambda: correlated_data(0.99, n), 200)
        block = max(n // 100, 1)   # 100 blocks
        print(f"{n:8,d} {sd:10.4f} {bootstrap_sd(x, y):9.4f} {bootstrap_sd(x, y, block):11.4f}")


if __name__ == "__main__":
    report()
