"""Week 11 reflection -- event studies on simulated trends.

Q1  The level, slope, and curvature all jump at the event. Compare a model
    that only looks for a level jump with one that looks for all three.
Q2  Three groups with different baselines jump by the same amount at a shared
    event. Group fixed effects + an event study recover the common effect.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

rng = np.random.default_rng(0)


# --- Q1: value, derivative, and second derivative all change ----------------

LEVEL, SLOPE, CURVE = 2, 0.1, 0.002      # true jumps after the event


def trend_data(n=200, event=100, effect=True):
    s = np.arange(n) - event              # time measured from the event
    post = (s >= 0).astype(int)
    y = 10 + 0.05 * s + rng.normal(0, 1, n)
    if effect:
        y += post * (LEVEL + SLOPE * s + CURVE * s**2)
    return pd.DataFrame({"y": y, "s": s, "post": post})


def value_only(df):
    """(a) a single level jump on top of a straight-line trend."""
    fit = smf.ols("y ~ s + post", df).fit()
    return fit.params["post"], fit.pvalues["post"] < 0.05


def all_three(df):
    """(b) jumps in level, slope, and curvature, tested together with an F-test."""
    fit = smf.ols("y ~ s + post + post:s + post:I(s**2)", df).fit()
    joint = fit.f_test("post = 0, post:s = 0, post:I(s ** 2) = 0")
    return fit.params["post"], joint.pvalue < 0.05


def compare(n_sims=1_000):
    rows = []
    for name, model in [("(a) value only", value_only), ("(b) all three", all_three)]:
        real = np.array([model(trend_data()) for _ in range(n_sims)])
        null = np.array([model(trend_data(effect=False)) for _ in range(n_sims)])
        rows.append({"model": name,
                     "mean level est": real[:, 0].mean(),
                     "sd level est": real[:, 0].std(ddof=1),
                     "detected (real)": real[:, 1].mean(),
                     "false pos (null)": null[:, 1].mean()})
    return pd.DataFrame(rows).set_index("model").round(3)


# --- Q2: three groups, parallel trends, one shared event --------------------

def group_data(n=100, event=50, jump=3):
    t = np.arange(n)
    frames = [pd.DataFrame({"group": g, "t": t, "post": (t >= event).astype(int),
                            "y": base + 0.1 * t + jump * (t >= event) + rng.normal(0, 1, n)})
              for g, base in zip("ABC", [0, 5, 10])]
    return pd.concat(frames, ignore_index=True)


def report():
    print(f"Q1 true jumps: level {LEVEL}, slope {SLOPE}, curvature {CURVE}")
    df = trend_data()
    fit_a = smf.ols("y ~ s + post", df).fit()
    fit_b = smf.ols("y ~ s + post + post:s + post:I(s**2)", df).fit()
    print("  (a) one dataset:", fit_a.params.round(3).to_dict())
    print("  (b) one dataset:", fit_b.params.round(4).to_dict())
    print("\n  1,000 datasets each:")
    print(compare())

    df = group_data()
    pooled = smf.ols("y ~ t + post", df).fit()
    fe = smf.ols("y ~ C(group) + t + post", df).fit()
    print("\nQ2 true jump 3, baselines 0 / 5 / 10")
    print(f"  no group FE:   post = {pooled.params['post']:.3f} (SE {pooled.bse['post']:.3f})")
    print(f"  with group FE: post = {fe.params['post']:.3f} (SE {fe.bse['post']:.3f})")
    print("  group effects:", fe.params.filter(like="group").round(2).to_dict(),
          f"(group A intercept {fe.params['Intercept']:.2f})")


if __name__ == "__main__":
    report()
