"""Week 3 -- regression discontinuity (homework_3.1.csv), differences-in-differences
(homework_3.2.a.csv, homework_3.2.b.csv).

Answers: Q1 B, Q2 C, Q3 A, Q4 A, Q5 B.
"""

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

CSVS = Path(__file__).parent / "csvs"
SERIES = ["value1", "value2", "value3"]
EVENT = 50


def load(name):
    return pd.read_csv(CSVS / name)


def best(scores):
    return max(scores, key=scores.get)


# --- Q1-Q2: discontinuity at time = 50 ------------------------------------

def report_discontinuity():
    df = load("homework_3.1.csv")
    df["post"] = (df["time"] >= EVENT).astype(int)
    df["s"] = df["time"] - EVENT  # centered, so "post" is the jump at time 50

    # Value only: y ~ time + post. Value and slope: y ~ s + post + s:post.
    jump = {v: smf.ols(f"{v} ~ time + post", df).fit().tvalues["post"] for v in SERIES}
    slope = {v: smf.ols(f"{v} ~ s * post", df).fit().tvalues["s:post"] for v in SERIES}

    for v in SERIES:
        print(f"{v}: jump t = {jump[v]:5.2f}   slope change t = {slope[v]:5.2f}")

    # Q1: value3 jumps ~1.77 (t = 4.4); value1/value2 jumps are not significant.
    # Q2: value1's slope rises 0.105 after the event (t = 7.9).
    print(f"\nQ1 strongest value discontinuity: {best(jump)}  -> B (value3)")
    print(f"Q2 strongest slope discontinuity: {best(slope)}  -> C (value1)")


# --- Q3-Q5: differences-in-differences ------------------------------------

def diff_in_diff(name):
    """outcome ~ group + time + group:time; the interaction is the treatment effect."""
    df = load(name)
    df.columns = ["group", "time", "outcome"]
    return smf.ols("outcome ~ group * time", df).fit()


def report_diff_in_diff():
    fits = {"Group 1": diff_in_diff("homework_3.2.a.csv"),
            "Group 2": diff_in_diff("homework_3.2.b.csv")}
    effect = {g: fit.params["group:time"] for g, fit in fits.items()}
    t = {g: fit.tvalues["group:time"] for g, fit in fits.items()}

    print()
    for g in fits:
        print(f"{g}: effect = {effect[g]:.4f}   t = {t[g]:.2f}")

    # Q3: Group 2's effect is about double Group 1's.
    # Q4: Group 2 is noisier, so Group 1 has the larger t despite the smaller effect.
    # Q5: 1.3499; of the options 1.248 is closest.
    print(f"\nQ3 largest treatment effect: {best(effect)}  -> A (Group 2)")
    print(f"Q4 most significant effect:  {best(t)}  -> A (Group 1)")
    print(f"Q5 Group 2 treatment effect: {effect['Group 2']:.4f}  -> B (1.248)")


if __name__ == "__main__":
    report_discontinuity()
    report_diff_in_diff()
