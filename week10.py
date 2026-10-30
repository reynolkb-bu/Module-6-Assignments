"""Week 10 -- time and city fixed effects (homework_10.1.csv), a log-transformed
outcome, and the reported vs simulated standard error under heteroskedasticity.

Answers: Q1 C, Q2 B, Q3 C, Q4 B.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

CSVS = Path(__file__).parent / "csvs"
DATA = CSVS / "homework_10.1.csv"
SEED = 0
rng = np.random.default_rng(SEED)


def load(path=DATA):
    return pd.read_csv(path)


# --- Q1-Q2: fixed effects for each time and each city -----------------------

def fixed_effects(df, group, other):
    """One constant per level of `group`, from y ~ X with dummies for both.

    Dropping the shared intercept (- 1) gives every level of `group` its own
    constant; `other` keeps a dummy for each level but its first.
    """
    fit = smf.ols(f"y ~ X + C({group}) + C({other}) - 1", df).fit()
    return fit.params.filter(like=f"C({group})")


def report_fixed_effects():
    df = load()
    print("Q1 time fixed effects:")
    print(fixed_effects(df, "time", "city").round(4).to_string())
    # 7.1 at month 0, peaks at 13.2 in month 5, back down to 8.9 by month 11.
    print("  -> C (rise to month 5 or 6, then fall)")
    print("\nQ2 city fixed effects:")
    print(fixed_effects(df, "city", "time").round(4).to_string())
    # Jumps around with no trend: 7.1, 1.4, 4.0, 9.0, 7.4, -3.9, 3.8, -0.6, -0.5, 1.6.
    print("  -> B (vary randomly)")


# --- Q3: modeling exp(Y) ----------------------------------------------------

def report_exp_y(num=1_000_000):
    X = np.clip(rng.normal(3, 1, num), 0.01, 100)
    Z = np.clip(rng.normal(3, 1, num), 0.01, 100)
    Y = np.log(X + Z) + rng.normal(0, 1, num)

    fit = sm.OLS(np.exp(Y), sm.add_constant(np.column_stack([X, Z]))).fit()
    # exp(Y) = (X + Z) * exp(noise), and E[exp(noise)] = exp(1/2) = 1.6487 for
    # standard normal noise, so both slopes converge to 1.65.
    print(f"\nQ3 {num:,} rows: coef of X = {fit.params[1]:.4f}, coef of Z = {fit.params[2]:.4f}"
          f"  (exp(0.5) = {np.exp(0.5):.4f})  -> C (both 1.6)")


# --- Q4: reported SE vs spread of 100 simulated estimates -------------------

def one_trial(num=10_000):
    """(coef of X, its reported SE) for one dataset from the quiz's DGP."""
    Z = rng.normal(0, 1, num)
    X = Z + rng.normal(0, 1, num)
    Y = 1.5 * X + 2.3 * Z + rng.normal(0, X**2, num)
    fit = sm.OLS(Y, sm.add_constant(np.column_stack([X, Z]))).fit()
    return fit.params[1], fit.bse[1]


def report_standard_error(n_trials=100):
    runs = np.array([one_trial() for _ in range(n_trials)])
    reported = runs[:, 1].mean()
    simulated = runs[:, 0].std(ddof=1)
    # The noise's sd is X^2, so the biggest errors sit at extreme X -- exactly
    # the rows that pull hardest on the slope. OLS's SE assumes one shared
    # variance and misses that, so the real spread is larger.
    print(f"\nQ4 {n_trials} datasets: (1) reported SE {reported:.4f}, "
          f"(2) sd of estimates {simulated:.4f}, ratio {simulated / reported:.2f}  -> B")


def report():
    if DATA.exists():
        report_fixed_effects()
    else:
        print(f"Q1-Q2 skipped: {DATA} not found")
    report_exp_y()
    report_standard_error()


if __name__ == "__main__":
    report()
