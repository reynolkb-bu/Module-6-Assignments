"""Week 9 -- how often Y ~ X + W detects X (|t| > 1.96), simulated from the quiz's DGP.

Answers: Q1 A, Q2 A, Q3 B, Q4 D.
"""

import numpy as np
import statsmodels.api as sm
from scipy.stats import skew

rng = np.random.default_rng(0)


def simulate(A=1, B=1, C=10, D=1000):
    """The quiz's data-generating code, drawing from a seeded rng."""
    W = rng.normal(0, 1, D)
    X = W + rng.normal(0, B, D)
    Y = A * X - W + rng.normal(0, C, D)
    return Y, X, W


def power(n_trials, **dgp):
    """Share of datasets where X is detected, plus every estimated coef of X."""
    coefs, detected = [], []
    for _ in range(n_trials):
        Y, X, W = simulate(**dgp)
        fit = sm.OLS(Y, sm.add_constant(np.column_stack([X, W]))).fit()
        coefs.append(fit.params[1])
        detected.append(abs(fit.tvalues[1]) > 1.96)
    return np.mean(detected), np.array(coefs)


def report():
    # 10,000 runs so the skew's own noise (~0.02) can't reach 0.15.
    detected, coefs = power(10_000)
    print(f"Q1 detected {detected:.1%}  -> A (88%)")
    print(f"Q2 skew of coef {skew(coefs):+.4f}  -> A (0)")

    # SE of X = C / (sqrt(D) * B). Power is 50% when A / SE = 1.96.
    print("\nQ3 A = 1, C = 10, D = 1000:")
    for B in [0.2, 0.6, 1.8, 5.4]:
        print(f"  B = {B}: detected {power(2_000, B=B)[0]:.1%}")
    print("  -> B (0.6)")

    print("\nQ4 B = 1, C = 10, D = 100:")
    for A in [0.5, 1.0, 2.0, 4.0]:
        print(f"  A = {A}: detected {power(2_000, A=A, D=100)[0]:.1%}")
    print("  -> D (2.0)")


if __name__ == "__main__":
    report()
