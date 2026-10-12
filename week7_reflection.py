"""Week 7 reflection -- leaving out a confounder, and running a test 1,000 times.

Q1  W causes X and Y. True effect of X on Y is 1.0.
Q2  W has no effect on Y. Fit Y ~ X + W 1,000 times and track W's p-value.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

N = 1_000
rng = np.random.default_rng(0)


def confounder(w_on_y):
    """Coef of X with W left out vs. with W controlled."""
    w = rng.normal(0, 1, N)
    x = w + rng.normal(0, 1, N)
    y = 1.0 * x + w_on_y * w + rng.normal(0, 1, N)
    df = pd.DataFrame({"W": w, "X": x, "Y": y})
    left_out = smf.ols("Y ~ X", df).fit().params["X"]
    controlled = smf.ols("Y ~ X + W", df).fit().params["X"]
    return left_out, controlled


def p_value_of_w():
    w = rng.normal(0, 1, N)
    x = rng.normal(0, 1, N)
    y = 2 * x + rng.normal(0, 1, N)
    df = pd.DataFrame({"W": w, "X": x, "Y": y})
    return smf.ols("Y ~ X + W", df).fit().pvalues["W"]


def report():
    print("Q1 true effect of X = 1.0")
    for w_on_y in [2.0, -2.0]:
        left_out, controlled = confounder(w_on_y)
        print(f"  W -> Y = {w_on_y:+.0f}:  Y ~ X {left_out:.4f}   Y ~ X + W {controlled:.4f}")

    p = np.array([p_value_of_w() for _ in range(N)])
    print(f"\nQ2 smallest p-value of W: {p.min():.6f}")
    print(f"   runs with p < 0.05: {(p < 0.05).sum()} of {N}")


if __name__ == "__main__":
    report()
