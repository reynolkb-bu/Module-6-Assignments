"""Week 5 -- a collider. X = trail difficulty, Y = speed, Z = accident likelihood.

X lowers Y (riders slow down on hard trails), and X and Y both raise Z. Z is a
collider on the path X -> Z <- Y, so the regression that controls for Z opens a
path that should stay closed.

Answers: Q1 C, Q2 D.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

NUM = 100_000         # rows per dataset
N_EXPERIMENTS = 200   # new dataset each time
SEED = 0


def simulate(rng, num=NUM):
    """The quiz's data-generating code, drawing from rng so a run can be repeated."""
    difficulty = rng.uniform(0, 1, (num,))
    speed = np.maximum(rng.normal(15, 5, (num,)) - difficulty * 10, 0)
    accident = np.minimum(np.maximum(
        0.03 * speed + 0.4 * difficulty + rng.normal(0, 0.3, (num,)), 0), 1)
    return pd.DataFrame({"difficulty": difficulty, "speed": speed, "accident": accident})


def coef_of_x(df, features):
    """Coefficient on difficulty from speed ~ const + features."""
    return sm.OLS(df["speed"], sm.add_constant(df[features])).fit().params["difficulty"]


def run_experiments(n=N_EXPERIMENTS, seed=SEED):
    rng = np.random.default_rng(seed)
    alone, with_z = np.empty(n), np.empty(n)
    for i in range(n):
        df = simulate(rng)
        alone[i] = coef_of_x(df, ["difficulty"])
        with_z[i] = coef_of_x(df, ["difficulty", "accident"])
    return alone, with_z


def report():
    alone, with_z = run_experiments()
    print(f"{'regression':<22} {'average coef of X':>18} {'std dev':>9}")
    print(f"{'speed ~ X':<22} {alone.mean():18.4f} {alone.std(ddof=1):9.4f}")
    print(f"{'speed ~ X + Z':<22} {with_z.mean():18.4f} {with_z.std(ddof=1):9.4f}")

    # Q1: -9.660. Not -10 because speed is floored at 0: on the hardest trails
    # some riders would be below 0 mph, and the floor pulls the slope up. The
    # std dev across datasets is ~0.05, so -9.661 is clearly the pick.
    # Q2: -10.322. Adding Z pushes the coefficient further from the true effect.
    print(f"\nQ1 average coef of X (X only):  {alone.mean():.4f}  -> C (-9.661)")
    print(f"Q2 average coef of X (X and Z): {with_z.mean():.4f}  -> D (-10.33)")

    # Should we control for Z? No. Z is a collider, so it is not a confounder
    # and there is no backdoor path to block. Among riders with the same accident
    # likelihood, a harder trail means the rider must have been going slower to
    # land on that same Z -- a link between X and Y that has nothing to do with
    # riders choosing to slow down. That link adds to the real effect and
    # overstates it by about 0.66 mph. Speed ~ X alone is the causal effect.
    print(f"\nbias from controlling for the collider Z: {with_z.mean() - alone.mean():.4f}")


if __name__ == "__main__":
    report()
