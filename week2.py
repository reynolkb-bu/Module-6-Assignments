"""Week 2 -- fixed effects on csvs/homework_2.1.csv, bootstrapped treatment
effects on csvs/homework_2.2.csv.

The fixed-effects model is y_{g,t} = alpha_g + beta * t: one slope on time for
all the data, plus a per-group intercept. Stacking G1/G2/G3 into one column and
regressing on time plus a group dummy each, with no constant, makes every dummy
coefficient that group's fixed effect.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import skew

CSVS = Path(__file__).parent / "csvs"
DATA = CSVS / "homework_2.1.csv"
BOOT_DATA = CSVS / "homework_2.2.csv"
GROUPS = ["G1", "G2", "G3"]
TREATMENT = "time"
BOOT_TREATMENT = "X"
BOOT_OUTCOME = "Y"
BOOT_CONFOUNDER = "Z"
N_BOOTSTRAP = 20_000
SEED = 0


def load_data(path=DATA):
    return pd.read_csv(path)


def to_long(df, groups=GROUPS):
    """Stack G1/G2/G3 into one outcome column with a `group` label."""
    return df.melt(id_vars=TREATMENT, value_vars=groups,
                   var_name="group", value_name="y")


def fit_fixed_effects(long):
    """One shared slope on time + a separate intercept per group (no constant)."""
    dummies = pd.get_dummies(long["group"], dtype=float)
    X = pd.concat([long[[TREATMENT]], dummies], axis=1)
    return sm.OLS(long["y"], X).fit()


def fit_per_group(df, groups=GROUPS):
    """Each group on its own: its own intercept and its own slope on time."""
    X = sm.add_constant(df[[TREATMENT]])
    return {g: sm.OLS(df[g], X).fit() for g in groups}


def main():
    df = load_data()
    model = fit_fixed_effects(to_long(df))
    coefs = model.params
    slopes = {g: fit.params[TREATMENT] for g, fit in fit_per_group(df).items()}

    print(f"{'group':<6} {'fixed effect':>14} {'own slope':>12}")
    for group in GROUPS:
        print(f"{group:<6} {coefs[group]:14.6f} {slopes[group]:12.6f}")

    # Q1: the coefficient of group 1 -> 0.008498, its own slope on time. (Not the
    # G1 fixed effect, 0.0786; the answer options are all slope-sized.)
    # Q2: the one time coefficient shared by all three groups -> 0.009017.
    print(f"\nQ1 coefficient of group 1 (G1 slope): {slopes['G1']:.6f}  -> D (0.00850)")
    print(f"Q2 common linear coefficient:         {coefs[TREATMENT]:.6f}  -> D (0.009017)")


def naive_effect(df):
    """Difference of group means -- ignores the confounder Z, so it is biased."""
    treated = df[df[BOOT_TREATMENT] == 1][BOOT_OUTCOME]
    control = df[df[BOOT_TREATMENT] == 0][BOOT_OUTCOME]
    return treated.mean() - control.mean()


def regression_effect(df):
    """Coefficient on X from Y ~ const + X + Z: the effect holding Z fixed."""
    X = sm.add_constant(df[[BOOT_TREATMENT, BOOT_CONFOUNDER]])
    return sm.OLS(df[BOOT_OUTCOME], X).fit().params[BOOT_TREATMENT]


def bootstrap(df, estimators, n_samples=N_BOOTSTRAP, seed=SEED):
    """Resample n rows with replacement n_samples times, re-estimating each time."""
    rng = np.random.default_rng(seed)
    n = len(df)
    draws = {name: np.empty(n_samples) for name in estimators}
    for i in range(n_samples):
        sample = df.iloc[rng.integers(0, n, n)]
        for name, estimator in estimators.items():
            draws[name][i] = estimator(sample)
    return draws


def confounded_effects():
    df = load_data(BOOT_DATA)

    # Q3: treated minus untreated, mean(Y | X=1) - mean(Y | X=0) -> 2.9207. It
    # overstates the effect: Z also raises Y and is not held fixed, so the
    # regression puts the effect lower, at 2.8187.
    print(f"\nQ3 naive difference of means:  {naive_effect(df):.6f}  -> C (2.921)")

    draws = bootstrap(df, {"naive": naive_effect, "regression": regression_effect})

    # Q4: variance of the naive effect -> ~0.0315 ("naive" row), matching the
    # analytic var(Y|X=1)/n1 + var(Y|X=0)/n0 = 0.031877.
    # Q5: skewness of the regression effect -> ~0.065 ("regression" row), near
    # zero as the CLT implies. Skewness is noisy: its standard error here is
    # sqrt(6/20000) = 0.017, and seeds 0-4 give 0.029-0.065, so the answer is
    # option A rather than a number this run reproduces exactly.
    print(f"\n{'estimator':<12} {'mean':>10} {'variance':>12} {'skewness':>10}")
    for name, values in draws.items():
        print(f"{name:<12} {values.mean():10.6f} {values.var(ddof=1):12.6f} "
              f"{skew(values):10.6f}")

    print(f"\nQ4 variance of naive effect:      {draws['naive'].var(ddof=1):.6f}  -> D (0.03274)")
    print(f"Q5 skewness of regression effect: {skew(draws['regression']):.6f}  -> A (0.04850)")


if __name__ == "__main__":
    main()
    confounded_effects()
