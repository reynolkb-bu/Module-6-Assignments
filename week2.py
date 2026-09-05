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
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from {path.name}")
    print(df.head(), "\n")
    print(df.describe(), "\n")
    print("Missing values per column:")
    print(df.isna().sum(), "\n")
    return df


def to_long(df, groups=GROUPS):
    """Stack G1/G2/G3 into one outcome column with a `group` label."""
    long = df.melt(id_vars=TREATMENT, value_vars=groups,
                   var_name="group", value_name="y")
    print(f"Reshaped to long: {long.shape[0]} rows "
          f"({df.shape[0]} times x {len(groups)} groups)")
    print(long.head(), "\n")
    return long


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
    long = to_long(df)
    model = fit_fixed_effects(long)
    print(model.summary())

    coefs = model.params
    print("\nFixed effects (group intercepts):")
    for group in GROUPS:
        print(f"  {group}: {coefs[group]:.6f}")

    # Q2: the one time coefficient shared by all three groups -> 0.009017.
    print(f"\nCommon linear coefficient on {TREATMENT}: {coefs[TREATMENT]:.6f}")
    print(f"R-squared: {model.rsquared:.4f}")

    # Q1: the coefficient of group 1 -> 0.008498, its own slope on time. (Not the
    # G1 fixed effect above, 0.0786; the options are all slope-sized.)
    print("\nPer-group regressions (own intercept, own slope):")
    print(f"{'group':<6} {'intercept':>12} {'slope':>12}")
    for group, fit in fit_per_group(df).items():
        print(f"{group:<6} {fit.params['const']:12.6f} {fit.params[TREATMENT]:12.6f}")


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
    treated = df[df[BOOT_TREATMENT] == 1]
    control = df[df[BOOT_TREATMENT] == 0]
    print(f"{len(treated)} treated rows, {len(control)} control rows")
    print(f"corr(X, Z) = {df[BOOT_TREATMENT].corr(df[BOOT_CONFOUNDER]):.4f}, "
          f"corr(Y, Z) = {df[BOOT_OUTCOME].corr(df[BOOT_CONFOUNDER]):.4f}\n")

    # Q3: treated minus untreated, mean(Y | X=1) - mean(Y | X=0) -> 2.9207. It
    # overstates the effect: Z also raises Y and is not held fixed, so the
    # regression puts the effect lower, at 2.8187.
    print(f"Point estimates on the full sample:")
    print(f"  naive difference of means: {naive_effect(df):.6f}")
    print(f"  regression Y ~ X + Z:      {regression_effect(df):.6f}\n")

    estimators = {"naive": naive_effect, "regression": regression_effect}
    print(f"Bootstrapping {N_BOOTSTRAP} resamples of {len(df)} rows (seed {SEED})...")
    draws = bootstrap(df, estimators)

    # Q4: variance of the naive effect -> ~0.0315 ("naive" row), matching the
    # analytic var(Y|X=1)/n1 + var(Y|X=0)/n0 = 0.031877.
    # Q5: skewness of the regression effect -> ~0.040 ("regression" row), near
    # zero as the CLT implies. Skewness is noisy: its standard error here is
    # sqrt(6/20000) = 0.017, so a different seed moves it a few hundredths.
    print(f"\n{'estimator':<12} {'mean':>10} {'variance':>12} {'skewness':>10}")
    for name, values in draws.items():
        print(f"{name:<12} {values.mean():10.6f} {values.var(ddof=1):12.6f} "
              f"{skew(values):10.6f}")


if __name__ == "__main__":
    main()
    confounded_effects()
