"""Week 12 -- differences-in-differences (homework_12.1.csv), a prior-trends
test (homework_12.2.csv), autocorrelated errors and the OLS standard error,
and the variance inflation factor.

DiD design: Group = 1 is treated, Time > 0 is the treatment period. The effect
is the Group x post interaction -- how much more the treated group changed
than the control group. It is causal only if the two groups would have moved
in parallel without treatment (prior trends).

Answers: Q1 B, Q2 C, Q3 D, Q4 A.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

CSVS = Path(__file__).parent / "csvs"
rng = np.random.default_rng(0)


# --- Q1: differences-in-differences ------------------------------------------

def report_did():
    df = pd.read_csv(CSVS / "homework_12.1.csv")
    df["post"] = (df["Time"] > 0).astype(int)
    fit = smf.ols("Y ~ Group * post", df).fit()
    print(f"Q1 DiD effect (Group:post) = {fit.params['Group:post']:.3f}"
          f" (SE {fit.bse['Group:post']:.3f})  -> B (1)")


# --- Q2: prior-trends test ---------------------------------------------------

def report_prior_trends():
    df = pd.read_csv(CSVS / "homework_12.2.csv")
    pre = df[df["Time"] < 0]
    # If the groups trend in parallel before treatment, Group:Time should be ~0.
    fit = smf.ols("Y ~ Group * Time", pre).fit()
    print(f"\nQ2 pre-period Group:Time = {fit.params['Group:Time']:.3f},"
          f" t = {fit.tvalues['Group:Time']:.2f}  -> C (5)")


# --- Q3: autocorrelated error ------------------------------------------------

RHO = 0.8
NUM = 10_000


def ar1_error():
    """AR(1): each value is RHO times the last plus fresh noise. Scaling the
    noise by sqrt(1 - RHO**2) keeps the sd at 1; starting from N(0, 1) keeps
    it there from the first step."""
    noise = rng.normal(size=NUM)
    err = np.empty(NUM)
    err[0] = noise[0]
    for t in range(1, NUM):
        err[t] = RHO * err[t - 1] + np.sqrt(1 - RHO**2) * noise[t]
    return err


def one_trial():
    X = ar1_error()
    Y = 2 * X + ar1_error()
    fit = sm.OLS(Y, sm.add_constant(X)).fit()
    return fit.params[1], fit.bse[1]


def report_autocorrelation(n_trials=300):
    err = ar1_error()
    print(f"\nQ3 error check: lag-1 corr {np.corrcoef(err[:-1], err[1:])[0, 1]:.3f},"
          f" mean {err.mean():.3f}, sd {err.std():.3f}")
    coefs, ses = np.array([one_trial() for _ in range(n_trials)]).T
    # OLS assumes independent rows. With both X and the error autocorrelated,
    # the true variance is (1 + RHO**2) / (1 - RHO**2) = 4.6x bigger, so the
    # simulated SE is about 2.1x the reported one.
    sim_se = coefs.std(ddof=1)
    print(f"  OLS SE {ses.mean():.4f}, simulated SE {sim_se:.4f}"
          f" (ratio {sim_se / ses.mean():.2f})  -> D (0.02)")


# --- Q4: variance inflation factor -------------------------------------------

def report_vif():
    np.random.seed(0)
    X1 = np.random.normal(0, 1, 1000)
    X2 = np.random.normal(0, 1, 1000) + X1
    X3 = np.random.normal(0, 1, 1000) + 2 * X2
    # VIF = 1 / (1 - R^2) of X1 on X2 and X3. X3 adds nothing about X1 beyond
    # X2, and X1 explains half of X2's variance, so R^2 ~ 0.5 and VIF ~ 2.
    design = sm.add_constant(np.column_stack([X1, X2, X3]))
    print(f"\nQ4 VIF of X1 = {variance_inflation_factor(design, 1):.2f}  -> A (2)")


def report():
    report_did()
    report_prior_trends()
    report_autocorrelation()
    report_vif()


if __name__ == "__main__":
    report()
