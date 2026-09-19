"""Week 4 -- instrumental variables (homework_4.1.csv), regression discontinuity
on college admission (homework_4.2.a.csv, homework_4.2.b.csv).

Answers: Q1 C, Q2 B, Q3 B, Q4 A, Q5 B.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

CSVS = Path(__file__).parent / "csvs"
CUTOFF = 80
W_GROUPS = 20


def load(name):
    return pd.read_csv(CSVS / name)


# --- Q1-Q2: Z is the instrument for the effect of X on Y --------------------

def iv_effect(df):
    """(change in Y from Z=0 to Z=1) / (change in X from Z=0 to Z=1)."""
    means = df.groupby("Z")[["X", "Y"]].mean()
    change = means.loc[1] - means.loc[0]
    return change["Y"] / change["X"]


def report_instrument():
    df = load("homework_4.1.csv")

    # Way 1: ignore W and use everyone at once.
    overall = iv_effect(df)
    # Way 2: split into groups with similar W, get the effect in each, and average.
    w_groups = pd.qcut(df["W"], W_GROUPS)
    by_w = np.mean([iv_effect(group) for _, group in df.groupby(w_groups, observed=True)])

    print(f"Effect ignoring W:        {overall:.4f}")
    print(f"Effect averaged over W:   {by_w:.4f}  ({W_GROUPS} groups)")

    # Q1: 1.5619, closest to 1.5.
    # Q2: Z is unrelated to W (corr 0.025), so both ways give ~1.5 -- W is not needed.
    print(f"\nQ1 effect (first way):  {overall:.4f}  -> C (1.5)")
    print("Q2 need W?  both ways give ~1.5  -> B (No)")


# --- Q3-Q5: discontinuity at a test score of 80 ----------------------------

def discontinuity(name):
    """Y ~ s + post + s:post, with s centered at the cutoff so post is the jump."""
    df = load(name)
    df.columns = ["X", "Y"]
    df["s"] = df["X"] - CUTOFF
    df["post"] = (df["X"] >= CUTOFF).astype(int)
    return smf.ols("Y ~ s * post", df).fit()


def report_discontinuity():
    print()
    for d in ["a", "b"]:
        fit = discontinuity(f"homework_4.2.{d}.csv")
        p, t = fit.params, fit.tvalues
        print(f"Dataset {d}: jump = {p['post']:.4f} (t = {t['post']:.1f})   "
              f"slope before = {p['s']:.4f} (t = {t['s']:.1f})   "
              f"slope after = {p['s'] + p['s:post']:.4f} (change t = {t['s:post']:.1f})")

    # Both datasets jump at 80 (+0.30 in a, +0.20 in b), so the math course helps in both.
    # Q3: a is flat on either side of 80 (slope t = 0.6); b clearly slopes (t = 36.3).
    # Q4: b's slope before the cutoff is +0.0102 per point.
    # Q5: after the cutoff it drops to 0.0050 (change t = -12.5).
    print("\nQ3 linear term: only dataset b has a real slope  -> B (Dataset b)")
    print("Q4 slope before cutoff in b: positive  -> A (Increasing)")
    print("Q5 slope after cutoff in b: smaller than before  -> B (Lower)")


if __name__ == "__main__":
    report_instrument()
    report_discontinuity()
