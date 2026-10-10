# Week 3 Reflection

## 1. In the event study in Coding Quiz 3, how would we go about testing for a change in the second derivative as well?

The best way I thought of to go about testing for a change in the second derivative is to think of a car. The value is where the car is at a certain point in time. The first derivative would be its speed, and the second derivative would be whether it's speeding up or slowing down.

The quiz tested whether the car suddenly moved to a new spot in time and whether its speed changed. Testing the second derivative is simply asking whether the car sped up or slowed down at a certain point in time. In order to test it we can add an additional term to the regression equation. The term would be a squared term that only applies after the event. If that term is different from zero, that means the curve changed at that event. 

I then ran this on the quiz data. In the Week 3 section of `Kyle_Reynolds_Weeks_1-4_Code.ipynb`, none of the three datasets actually showed a change in the second derivative, meaning the answer still holds. 

## 2. Create your own scenario that illustrates differences-in-differences. Describe the story behind the data and show whether there is a nonzero treatment effect.

My own scenario that illustrates differences in differences would be a software company that adds an onboarding checklist. They add this checklist because they're hoping to reduce the number of support tickets. U.S. customers get this onboarding checklist right away, but EU customers don't get it until later. 

Support tickets went down in the U.S. after the onboarding checklist launched, but they also went down in the EU, which never got the checklist. This means the drop in support tickets was not caused solely by the checklist.

In my simulated data, the number of U.S. support tickets dropped by 1.34 per month, and EU tickets dropped by 0.55 per month. The checklist did have a real effect: 1.34 - 0.55 = about 0.78 fewer support tickets per month. This means that there is a non-zero treatment effect since the onboarding checklist actually reduced the number of support tickets. 

```python
N_ACCOUNTS = 200          # per region
MONTHS = 12
LAUNCH = 7                # first month with the checklist
TRUE_EFFECT = -0.8        # tickets per account per month
SEED = 0

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
```
