"""Week 3 -- regression discontinuity (homework_3.1.csv), differences-in-differences
(homework_3.2.a.csv, homework_3.2.b.csv).

Discontinuity: with post = 1{time >= 50} and centered time s = time - 50,
    value-only:     y = a + b*time + c*post
    value + slope:  y = a + b*s + c*post + d*s*post
c is the jump in the value at the event, d the change in the slope.

Diff-in-diff: y = a + b*group + c*time + d*group*time, where d is the treatment
effect under parallel trends.

Answers: Q1 B, Q2 C, Q3 A, Q4 A, Q5 B.
"""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

CSVS = Path(__file__).parent / "csvs"
TIME = "time"
SERIES = ["value1", "value2", "value3"]
EVENT = 50
DID_FILES = {
    "Group 1": ("homework_3.2.a.csv", "group1", "time1", "outcome1"),
    "Group 2": ("homework_3.2.b.csv", "group2", "time2", "outcome2"),
}


def load(name):
    return pd.read_csv(CSVS / name)


# --- Q1-Q2: discontinuity at the event ------------------------------------

def value_jump(df, series):
    """Shared slope, intercept shifts at the event. Returns the fit."""
    post = (df[TIME] >= EVENT).astype(float)
    design = sm.add_constant(pd.DataFrame({TIME: df[TIME], "post": post}))
    return sm.OLS(df[series], design).fit()


def slope_jump(df, series):
    """Value and slope both free to change at the event. Time is centered on it,
    so "post" is still the jump at time = 50 rather than at time = 0."""
    post = (df[TIME] >= EVENT).astype(float)
    centered = df[TIME] - EVENT
    design = sm.add_constant(pd.DataFrame(
        {TIME: centered, "post": post, "time_post": centered * post}))
    return sm.OLS(df[series], design).fit()


def report_discontinuity():
    df = load("homework_3.1.csv")
    value_only = {s: value_jump(df, s) for s in SERIES}
    with_slope = {s: slope_jump(df, s) for s in SERIES}

    print(f"{'series':<7} {'jump':>8} {'t':>6} | {'jump':>8} {'t':>6} {'slope chg':>10} {'t':>6}")
    for s in SERIES:
        a, b = value_only[s], with_slope[s]
        print(f"{s:<7} {a.params['post']:8.4f} {a.tvalues['post']:6.2f} | "
              f"{b.params['post']:8.4f} {b.tvalues['post']:6.2f} "
              f"{b.params['time_post']:10.4f} {b.tvalues['time_post']:6.2f}")

    # Q1: value3 jumps ~1.77 (t = 4.4), about double value1/value2, whose jumps
    #     are not significant in the value-only model. Holds with the slope term too.
    # Q2: value1's slope rises by 0.105 after the event (t = 7.9); value2 and
    #     value3 change by 0.037 and 0.051.
    jump_t = {s: value_only[s].tvalues["post"] for s in SERIES}
    slope_t = {s: with_slope[s].tvalues["time_post"] for s in SERIES}
    print(f"\nQ1 strongest value discontinuity: {max(jump_t, key=jump_t.get)}  -> B (value3)")
    print(f"Q2 strongest slope discontinuity: {max(slope_t, key=slope_t.get)}  -> C (value1)")


# --- Q3-Q5: differences-in-differences ------------------------------------

def diff_in_diff(df, group, time, outcome):
    """Coefficient on group*time is the treatment effect."""
    design = sm.add_constant(pd.DataFrame(
        {"group": df[group], "time": df[time], "group_time": df[group] * df[time]}))
    return sm.OLS(df[outcome], design).fit()


def report_diff_in_diff():
    fits = {name: diff_in_diff(load(f), g, t, y) for name, (f, g, t, y) in DID_FILES.items()}

    print(f"\n{'dataset':<8} {'effect':>8} {'std err':>8} {'t':>7} {'p':>10}")
    for name, fit in fits.items():
        print(f"{name:<8} {fit.params['group_time']:8.4f} {fit.bse['group_time']:8.4f} "
              f"{fit.tvalues['group_time']:7.2f} {fit.pvalues['group_time']:10.2e}")

    effects = {name: fit.params["group_time"] for name, fit in fits.items()}
    t_stats = {name: fit.tvalues["group_time"] for name, fit in fits.items()}

    # Q3: Group 2's effect (1.350) is about double Group 1's (0.686).
    # Q4: Group 2 is noisier -- std err 0.147 vs 0.063 -- so Group 1 has the
    #     larger t (11.0 vs 9.2) despite the smaller effect.
    # Q5: 1.350; of the options 1.248 is closest.
    print(f"\nQ3 largest treatment effect:     {max(effects, key=effects.get)}  -> A (Group 2)")
    print(f"Q4 most significant effect:      {max(t_stats, key=t_stats.get)}  -> A (Group 1)")
    print(f"Q5 Group 2 treatment effect:     {effects['Group 2']:.4f}  -> B (1.248)")


if __name__ == "__main__":
    report_discontinuity()
    report_diff_in_diff()
