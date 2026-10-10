"""Week 7 reflection -- leaving out a confounder, and running a test 1,000 times.

Q1  W causes both X and Y, but the regression only uses X.
        W = noise
        X = W + noise
        Y = 1.0 X + 2.0 W + noise
    True effect of X on Y is 1.0. Y ~ X picks up part of W's effect too.
    A second version flips W's effect on Y to -2.0 to show the other direction.

Q2  W has no effect on Y at all.
        W = noise
        X = noise
        Y = 2 X + noise
    Fit Y ~ X + W, record the p-value of W, repeat 1,000 times.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

N_ROWS = 1_000
N_RUNS = 1_000
X_ON_Y = 1.0
SEED = 0
rng = np.random.default_rng(SEED)


# --- Q1: omitted confounder -------------------------------------------------

def simulate_confounded(w_on_y, n=N_ROWS):
    w = rng.normal(0, 1, n)
    x = w + rng.normal(0, 1, n)
    y = X_ON_Y * x + w_on_y * w + rng.normal(0, 1, n)
    return pd.DataFrame({"W": w, "X": x, "Y": y})


def report_confounder():
    print(f"Q1 true effect of X on Y = {X_ON_Y}")
    print(f"{'W -> Y':>7} {'Y ~ X':>8} {'Y ~ X + W':>10} {'expected bias':>14}")
    for w_on_y in [2.0, -2.0]:
        df = simulate_confounded(w_on_y)
        left_out = smf.ols("Y ~ X", df).fit().params["X"]
        controlled = smf.ols("Y ~ X + W", df).fit().params["X"]
        # Bias = (W -> Y) * cov(X, W) / var(X) = w_on_y * 1 / 2.
        print(f"{w_on_y:7.1f} {left_out:8.4f} {controlled:10.4f} {w_on_y / 2:+14.2f}")


# --- Q2: smallest p-value of a zero coefficient -----------------------------

def p_value_of_w(n=N_ROWS):
    w = rng.normal(0, 1, n)
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"W": w, "X": x, "Y": y})
    return smf.ols("Y ~ X + W", df).fit().pvalues["W"]


def report_p_values():
    p = np.array([p_value_of_w() for _ in range(N_RUNS)])
    print(f"\nQ2 p-value of W over {N_RUNS} runs (true coef of W = 0):")
    print(f"  smallest p-value:  {p.min():.6f}")
    print(f"  runs with p < 0.05: {(p < 0.05).sum()} of {N_RUNS} ({(p < 0.05).mean():.1%})")
    print(f"  median p-value:    {np.median(p):.4f}")


if __name__ == "__main__":
    report_confounder()
    report_p_values()
