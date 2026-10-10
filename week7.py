"""Week 7 -- omitted variables, controlling for W by slicing (homework_7.1.csv),
and autocorrelated errors.

Answers: Q1 B, Q2 E, Q3 B, Q4 A.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

CSVS = Path(__file__).parent / "csvs"
DATA = CSVS / "homework_7.1.csv"
N = 1_000
W_VALUES = [-1, 0, 1]
BAND = 0.1                     # rows with W within 0.1 of the target count as "W constant"
CORR_CONSTS = [0.2, 0.5, 0.8]
N_TRIALS = 1_000
N_ROWS = 100
SEED = 0
rng = np.random.default_rng(SEED)


def load(path=DATA):
    return pd.read_csv(path)


# --- Q1-Q2: correlation of X with the error term ----------------------------

def report_error_correlation():
    w = rng.normal(0, 1, N)
    x = w + rng.normal(0, 1, N)
    z = rng.normal(0, 1, N)
    noise = rng.normal(0, 1, N)
    y = x + z + w + noise

    # Q1: W is in the model, so the error is just the noise -- independent of X.
    # Q2: W is left out, so the error is W + noise. X and the error share W:
    #     cov = var(W) = 1, sd(X) = sd(W + noise) = sqrt(2) -> corr = 1/2.
    print(f"Q1 corr(X, noise):     {np.corrcoef(x, noise)[0, 1]:.4f}  -> B (0)")
    print(f"Q2 corr(X, W + noise): {np.corrcoef(x, w + noise)[0, 1]:.4f}  -> E (0.50)")


# --- Q3: coefficient of X at W = -1, 0, 1 -----------------------------------

def coef_of_x_near(df, w, band=BAND):
    """Regress Y ~ X + Z on the rows whose W is within `band` of w."""
    near = df[(df["W"] - w).abs() < band]
    return len(near), smf.ols("Y ~ X + Z", near).fit().params["X"]


def report_slices():
    df = load()
    # W is continuous, so no two rows share an exact value. Keeping only rows
    # with W close to the target holds W (nearly) constant inside each slice.
    print(f"\nQ3 Y ~ X + Z on rows with |W - w| < {BAND}:")
    for w in W_VALUES:
        rows, coef = coef_of_x_near(df, w)
        print(f"  W = {w:+d}: {rows:4d} rows, coef of X = {coef:.4f}")
    # 0.86 -> 1.38 -> 1.96: up by about 0.5 per unit of W -> B (increasing).
    print("  -> B (increasing)")


# --- Q4: autocorrelated error on both X and Y -------------------------------

def make_error(corr_const, num):
    err = list()
    prev = rng.normal(0, 1)
    for n in range(num):
        prev = corr_const * prev + (1 - corr_const) * rng.normal(0, 1)
        err.append(prev)
    return np.array(err)


def one_trial(corr_const, n=N_ROWS):
    """(estimated coef of X, its reported standard error) for one dataset."""
    x = make_error(corr_const, n)
    y = 1 + 2 * x + make_error(corr_const, n)
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    return fit.params[1], fit.bse[1]


def report_autocorrelation():
    print(f"\nQ4 {N_TRIALS} trials of {N_ROWS} rows, Y = 1 + 2X + error:")
    print(f"{'corr_const':>10} {'(i) sd of coef':>15} {'(ii) mean SE':>13} {'ratio':>7}")
    for c in CORR_CONSTS:
        runs = np.array([one_trial(c) for _ in range(N_TRIALS)])
        sd_coef = runs[:, 0].std(ddof=1)
        mean_se = runs[:, 1].mean()
        print(f"{c:10.1f} {sd_coef:15.4f} {mean_se:13.4f} {sd_coef / mean_se:7.3f}")
    # The reported SE stays near 0.10 but the real spread grows, because OLS
    # assumes each error is independent. -> A (the ratio increases).
    print("  -> A (ratio increases)")


if __name__ == "__main__":
    report_error_correlation()
    report_slices()
    report_autocorrelation()
