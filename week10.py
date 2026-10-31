"""Week 10 -- time and city fixed effects (homework_10.1.csv), a log-transformed
outcome, and the reported vs simulated standard error under heteroskedasticity.

Answers: Q1 C, Q2 B, Q3 C, Q4 B.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA = Path(__file__).parent / "csvs" / "homework_10.1.csv"
rng = np.random.default_rng(0)


def ols(y, *columns):
    return sm.OLS(y, sm.add_constant(np.column_stack(columns))).fit()


# --- Q1-Q2: fixed effects ---------------------------------------------------

def report_fixed_effects():
    df = pd.read_csv(DATA)
    # "- 1" drops the shared intercept, so the first dummy set listed gets one
    # constant per level -- those constants are the fixed effects.
    time_fe = smf.ols("y ~ X + C(time) + C(city) - 1", df).fit().params.filter(like="time")
    city_fe = smf.ols("y ~ X + C(city) + C(time) - 1", df).fit().params.filter(like="city")
    print("Q1 time fixed effects:", time_fe.round(1).tolist())
    print("  -> C (rise to month 5, then fall)")
    print("Q2 city fixed effects:", city_fe.round(1).tolist())
    print("  -> B (vary randomly)")


# --- Q3: modeling exp(Y) ----------------------------------------------------

def report_exp_y(num=1_000_000):
    X = np.clip(rng.normal(3, 1, num), 0.01, 100)
    Z = np.clip(rng.normal(3, 1, num), 0.01, 100)
    Y = np.log(X + Z) + rng.normal(0, 1, num)
    # exp(Y) = (X + Z) * exp(noise), and the mean of exp(noise) is exp(0.5) = 1.65.
    fit = ols(np.exp(Y), X, Z)
    print(f"\nQ3 coef of X = {fit.params[1]:.3f}, coef of Z = {fit.params[2]:.3f}  -> C (both 1.6)")


# --- Q4: reported SE vs spread of 100 simulated estimates -------------------

def one_trial(num=10_000):
    Z = rng.normal(0, 1, num)
    X = Z + rng.normal(0, 1, num)
    Y = 1.5 * X + 2.3 * Z + rng.normal(0, X**2, num)
    fit = ols(Y, X, Z)
    return fit.params[1], fit.bse[1]


def report_standard_error():
    coefs, ses = np.array([one_trial() for _ in range(100)]).T
    # The noise grows with X, but OLS's SE assumes it's the same everywhere,
    # so it understates how much the estimate really varies.
    print(f"\nQ4 (1) reported SE {ses.mean():.4f}, (2) sd of 100 estimates {coefs.std(ddof=1):.4f}  -> B")


if __name__ == "__main__":
    report_fixed_effects()
    report_exp_y()
    report_standard_error()
