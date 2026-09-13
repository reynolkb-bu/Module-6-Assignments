"""Week 3 reflection -- a second-derivative break on homework_3.1.csv, and an
invented differences-in-differences scenario.

Q1: add s^2 and s^2*post to the event-study regression. The s^2*post
coefficient is half the change in the second derivative at the event.

Q2: a SaaS company ships a new onboarding checklist to its US accounts in month
7. EU accounts wait on a privacy review, so they keep the old flow and serve as
the control. Outcome: support tickets per account per month.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

CSVS = Path(__file__).parent / "csvs"
TIME = "time"
SERIES = ["value1", "value2", "value3"]
EVENT = 50

N_ACCOUNTS = 200          # per region
MONTHS = 12
LAUNCH = 7                # first month with the checklist
TRUE_EFFECT = -0.8        # tickets per account per month
SEED = 0


# --- Q1: change in the second derivative ----------------------------------

def curvature_jump(df, series):
    """y = a + b*s + c*s^2 + d*post + e*s*post + f*s^2*post, s = time - 50.
    d, e, 2f are the jumps in value, slope, and second derivative at the event."""
    s = df[TIME] - EVENT
    post = (df[TIME] >= EVENT).astype(float)
    design = sm.add_constant(pd.DataFrame({
        "s": s, "s2": s**2, "post": post, "s_post": s * post, "s2_post": s**2 * post}))
    return sm.OLS(df[series], design).fit()


def compare_nested(df, series):
    """F-test: does adding both curvature terms improve on the slope-break model?"""
    s = df[TIME] - EVENT
    post = (df[TIME] >= EVENT).astype(float)
    small = sm.add_constant(pd.DataFrame({"s": s, "post": post, "s_post": s * post}))
    restricted = sm.OLS(df[series], small).fit()
    return curvature_jump(df, series).compare_f_test(restricted)


def report_second_derivative():
    df = pd.read_csv(CSVS / "homework_3.1.csv")
    print(f"{'series':<7} {'2f (d2 chg)':>12} {'t':>6} {'p':>7} | {'F':>6} {'p':>7}")
    for s in SERIES:
        fit = curvature_jump(df, s)
        f_stat, f_p, _ = compare_nested(df, s)
        print(f"{s:<7} {2 * fit.params['s2_post']:12.5f} {fit.tvalues['s2_post']:6.2f} "
              f"{fit.pvalues['s2_post']:7.3f} | {f_stat:6.2f} {f_p:7.3f}")


# --- Q2: invented diff-in-diff scenario -----------------------------------

def simulate(seed=SEED):
    """Tickets fall ~0.1/month in both regions as the product matures (parallel
    trends). US accounts start higher; the checklist cuts 0.8 more from launch."""
    rng = np.random.default_rng(seed)
    rows = []
    for us in (0, 1):
        baseline = 4.0 + 1.0 * us
        for account in range(N_ACCOUNTS):
            account_level = rng.normal(0, 1.0)          # some accounts just file more
            for month in range(1, MONTHS + 1):
                post = int(month >= LAUNCH)
                mean = baseline + account_level - 0.1 * month + TRUE_EFFECT * us * post
                rows.append({"account": f"{us}-{account}", "us": us, "month": month,
                             "post": post, "tickets": mean + rng.normal(0, 1.0)})
    return pd.DataFrame(rows)


def diff_in_diff(df):
    """tickets ~ us + post + us*post, SEs clustered by account (12 rows each)."""
    design = sm.add_constant(df[["us", "post"]].assign(us_post=df["us"] * df["post"]))
    return sm.OLS(df["tickets"], design).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df["account"])[0]})


def pre_trend(df):
    """Before launch only: does the US trend differ from the EU trend?"""
    pre = df[df["post"] == 0]
    design = sm.add_constant(pre[["us", "month"]].assign(us_month=pre["us"] * pre["month"]))
    return sm.OLS(pre["tickets"], design).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(pre["account"])[0]})


def report_scenario():
    df = simulate()
    means = df.groupby(["us", "post"])["tickets"].mean().unstack()
    means.index = ["EU (control)", "US (treated)"]
    means.columns = ["before", "after"]
    means["change"] = means["after"] - means["before"]
    print("\n", means.round(3), sep="")

    naive = means.loc["US (treated)", "change"]
    manual = naive - means.loc["EU (control)", "change"]
    fit = diff_in_diff(df)
    lo, hi = fit.conf_int().loc["us_post"]
    trend = pre_trend(df)

    print(f"\nnaive US before/after:    {naive:.4f}")
    print(f"diff-in-diff (by hand):   {manual:.4f}")
    print(f"diff-in-diff (regression): {fit.params['us_post']:.4f}  "
          f"se {fit.bse['us_post']:.4f}  t {fit.tvalues['us_post']:.2f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]  p {fit.pvalues['us_post']:.1e}")
    print(f"pre-launch trend gap:     {trend.params['us_month']:.4f}  "
          f"se {trend.bse['us_month']:.4f}  p {trend.pvalues['us_month']:.3f}")


if __name__ == "__main__":
    report_second_derivative()
    report_scenario()
