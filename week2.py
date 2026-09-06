"""Week 2 -- fixed effects (homework_2.1.csv), bootstrapped effects (homework_2.2.csv).

Fixed effects: y_{g,t} = alpha_g + beta * t -- one shared slope on time, one
intercept per group. Stack G1/G2/G3 into a single column, regress on time plus a
dummy per group with no constant, and each dummy coefficient is that group's
fixed effect.

Answers: Q1 D, Q2 D, Q3 C, Q4 D, Q5 A.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import skew

CSVS = Path(__file__).parent / "csvs"
GROUPS = ["G1", "G2", "G3"]
TIME, X, Y, Z = "time", "X", "Y", "Z"
N_BOOTSTRAP = 20_000
SEED = 0


def load(name):
    return pd.read_csv(CSVS / name)


# --- Q1-Q2: fixed effects -------------------------------------------------

def fixed_effects(df):
    """Shared slope on time + one intercept per group. Returns the coefficients."""
    long = df.melt(id_vars=TIME, value_vars=GROUPS, var_name="group", value_name="y")
    design = pd.concat([long[[TIME]], pd.get_dummies(long["group"], dtype=float)], axis=1)
    return sm.OLS(long["y"], design).fit().params


def own_slopes(df):
    """Each group fit alone: its own intercept and its own slope on time."""
    design = sm.add_constant(df[[TIME]])
    return {g: sm.OLS(df[g], design).fit().params[TIME] for g in GROUPS}


def report_fixed_effects():
    df = load("homework_2.1.csv")
    coefs, slopes = fixed_effects(df), own_slopes(df)

    print(f"{'group':<6} {'fixed effect':>14} {'own slope':>12}")
    for g in GROUPS:
        print(f"{g:<6} {coefs[g]:14.6f} {slopes[g]:12.6f}")

    # Q1 wants G1's own slope, not its fixed effect (0.0786) -- the options are
    # all slope-sized. Q2 is the one time coefficient shared by every group.
    print(f"\nQ1 coefficient of group 1 (G1 slope): {slopes['G1']:.6f}  -> D (0.00850)")
    print(f"Q2 common linear coefficient:         {coefs[TIME]:.6f}  -> D (0.009017)")


# --- Q3-Q5: confounding and the bootstrap ---------------------------------

def naive_effect(df):
    """Difference of group means. Ignores Z, so it is biased upward."""
    return df[df[X] == 1][Y].mean() - df[df[X] == 0][Y].mean()


def regression_effect(df):
    """Coefficient on X from Y ~ const + X + Z: the effect holding Z fixed."""
    return sm.OLS(df[Y], sm.add_constant(df[[X, Z]])).fit().params[X]


def bootstrap(df, estimators, n_samples=N_BOOTSTRAP, seed=SEED):
    """Resample n rows with replacement n_samples times, re-estimating each time."""
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(n_samples) for name in estimators}
    for i in range(n_samples):
        sample = df.iloc[rng.integers(0, len(df), len(df))]
        for name, estimate in estimators.items():
            draws[name][i] = estimate(sample)
    return draws


def report_bootstrap():
    df = load("homework_2.2.csv")

    # Q3 overstates the effect: Z raises Y too and is not held fixed, so the
    # regression lands lower, at 2.8187.
    print(f"\nQ3 naive difference of means:  {naive_effect(df):.6f}  -> C (2.921)")

    draws = bootstrap(df, {"naive": naive_effect, "regression": regression_effect})

    print(f"\n{'estimator':<12} {'mean':>10} {'variance':>12} {'skewness':>10}")
    for name, values in draws.items():
        print(f"{name:<12} {values.mean():10.6f} {values.var(ddof=1):12.6f} "
              f"{skew(values):10.6f}")

    # Q4 matches the analytic var(Y|X=1)/n1 + var(Y|X=0)/n0 = 0.031877.
    # Q5 is near zero, as the CLT implies, but skewness is noisy: its standard
    # error is sqrt(6/20000) = 0.017 and seeds 0-4 give 0.029-0.065. So the
    # answer is option A, not a number this run reproduces exactly.
    print(f"\nQ4 variance of naive effect:      {draws['naive'].var(ddof=1):.6f}  -> D (0.03274)")
    print(f"Q5 skewness of regression effect: {skew(draws['regression']):.6f}  -> A (0.04850)")


if __name__ == "__main__":
    report_fixed_effects()
    report_bootstrap()
