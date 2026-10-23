"""Week 9 -- power and skew of the coefficient of X, simulated from the quiz's DGP.

    W = N(0, 1)
    X = W + N(0, B)
    Y = A X - W + N(0, C)       D rows per dataset, Y ~ X + W every time

"Detected" means |t-value of X| > 1.96.

Answers: Q1 A, Q2 A, Q3 B, Q4 D.
"""

import numpy as np
import statsmodels.api as sm
from scipy.stats import skew

N_TRIALS = 10_000     # Q1-Q2: enough that the skew's own noise (~0.02) can't reach 0.15
N_SCAN = 2_000        # Q3-Q4: datasets per candidate value
T_CUTOFF = 1.96
B_OPTIONS = [0.2, 0.6, 1.8, 5.4]
A_OPTIONS = [0.5, 1.0, 2.0, 4.0]
SEED = 0
rng = np.random.default_rng(SEED)


def simulate(A=1, B=1, C=10, D=1000):
    """The quiz's data-generating code, drawing from a seeded rng."""
    W = rng.normal(0, 1, D)
    X = W + rng.normal(0, B, D)
    Y = A * X - W + rng.normal(0, C, D)
    return Y, X, W


def one_trial(**dgp):
    """(estimated coef of X, its t-value) from Y ~ X + W on one dataset."""
    Y, X, W = simulate(**dgp)
    fit = sm.OLS(Y, sm.add_constant(np.column_stack([X, W]))).fit()
    return fit.params[1], fit.tvalues[1]


def power(n_trials=N_TRIALS, **dgp):
    """Share of datasets where X is detected, plus every estimated coef."""
    runs = np.array([one_trial(**dgp) for _ in range(n_trials)])
    return (np.abs(runs[:, 1]) > T_CUTOFF).mean(), runs[:, 0]


# --- Q1-Q2: A = 1, B = 1, C = 10, D = 1000 ----------------------------------

def report_baseline():
    detected, coefs = power()
    # With W in the model, X only varies through its own noise (sd B), so
    # SE = C / (sqrt(D) * B) = 10 / sqrt(1000) = 0.32 and t is about 1 / 0.32 = 3.2.
    print(f"Q1 {N_TRIALS} datasets, detected {detected:.1%} of the time  -> A (88%)")
    # OLS is unbiased here and the noise is normal, so the estimates are
    # symmetric around 1.
    print(f"Q2 mean coef {coefs.mean():.4f}, skew {skew(coefs):+.4f}  -> A (0)")


# --- Q3-Q4: find the value that gives about 50% power -----------------------

def report_scan(name, options, answer, **dgp):
    for value in options:
        detected, _ = power(N_SCAN, **dgp, **{name: value})
        print(f"  {name} = {value:3.1f}: detected {detected:6.1%}")
    print(f"  -> {answer}")


def report():
    report_baseline()

    # Q3: SE = 10 / (sqrt(1000) * B). Power is 50% when A / SE = 1.96, so B = 0.62.
    print(f"\nQ3 A = 1, C = 10, D = 1000, {N_SCAN} datasets per B:")
    report_scan("B", B_OPTIONS, "B (0.6)", A=1, C=10, D=1000)

    # Q4: D = 100 makes SE = 10 / (sqrt(100) * 1) = 1, so A must be about 1.96.
    print(f"\nQ4 B = 1, C = 10, D = 100, {N_SCAN} datasets per A:")
    report_scan("A", A_OPTIONS, "D (2.0)", B=1, C=10, D=100)


if __name__ == "__main__":
    report()
