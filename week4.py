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
W_BINS = 20


def load(name):
    return pd.read_csv(CSVS / name)


def wald(df):
    """(Y|Z=1 - Y|Z=0) / (X|Z=1 - X|Z=0)."""
    m = df.groupby("Z")[["X", "Y"]].mean()
    return (m.loc[1, "Y"] - m.loc[0, "Y"]) / (m.loc[1, "X"] - m.loc[0, "X"])


# --- Q1-Q2: instrument Z for the effect of X on Y ---------------------------

def report_instrument():
    df = load("homework_4.1.csv")

    # Way 1: one ratio over the whole sample, ignoring W.
    overall = wald(df)
    # Way 2: the same ratio within narrow ranges of W, then averaged.
    bins = pd.qcut(df["W"], W_BINS)
    by_w = np.mean([wald(s) for _, s in df.groupby(bins, observed=True)])

    print(f"Overall ratio:              {overall:.4f}")
    print(f"Average of ratios by W:     {by_w:.4f}  ({W_BINS} bins)")

    # Q1: 1.5619, closest to 1.5.
    # Q2: Z is independent of W (corr 0.025), so ignoring W gives the same answer
    #     as conditioning on it -- W is not needed.
    print(f"\nQ1 effect (first way):  {overall:.4f}  -> C (1.5)")
    print(f"Q2 need W?  both ways give ~1.5  -> B (No)")


# --- Q3-Q5: discontinuity at a test score of 80 ----------------------------

def discontinuity(name):
    """Y ~ s + post + s:post, with s centered at the cutoff so post is the jump."""
    df = load(name)
    df.columns = ["X", "Y"]
    df["s"] = df["X"] - CUTOFF
    df["post"] = (df["X"] >= CUTOFF).astype(int)
    return smf.ols("Y ~ s * post", df).fit()


def report_discontinuity():
    fits = {"a": discontinuity("homework_4.2.a.csv"),
            "b": discontinuity("homework_4.2.b.csv")}

    print()
    for d, fit in fits.items():
        p, t = fit.params, fit.tvalues
        print(f"Dataset {d}: jump = {p['post']:.4f} (t = {t['post']:5.1f})   "
              f"slope before = {p['s']:.4f} (t = {t['s']:5.1f})   "
              f"slope after = {p['s'] + p['s:post']:.4f} (change t = {t['s:post']:5.1f})")

    # Both datasets jump at 80 (+0.30 in a, +0.20 in b), so the math course helps in both.
    # Q3: a is flat on either side of 80 (slope t = 0.6); b clearly slopes (t = 36).
    # Q4: b's slope before the cutoff is +0.0102 per point.
    # Q5: after the cutoff it drops to 0.0050 (change t = -12.5).
    b = fits["b"].params
    print(f"\nQ3 linear term: dataset b slope t = {fits['b'].tvalues['s']:.1f}  -> B (Dataset b)")
    print(f"Q4 slope before cutoff: {b['s']:.4f}  -> A (Increasing)")
    print(f"Q5 slope after cutoff:  {b['s'] + b['s:post']:.4f}  -> B (Lower)")


if __name__ == "__main__":
    report_instrument()
    report_discontinuity()
