"""Week 6 reflection -- why the max untreated effect overstates the optimal
treatment effect, and a fix.

homework_6.1.csv has no noise at all: untreated Y = -1.5 Z and treated
Y = 1.2 - 0.5 Z exactly, so the true effect is 1.2 + Z. The simulation below
uses that same setup and adds noise to Y, so we know the true answer and can
see how far each estimate lands from it.

Three ways to estimate the optimal effect from the untreated rows' effects:
    max          the single largest effect (the quiz's Q4 method)
    90th pct     the 90th percentile of the effects
    smoothed max fit effect ~ Z on all untreated rows, then take the fitted
                 effect at the untreated row where the line is highest
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from week6 import load, unit_effects

N_ROWS = 1_000
N_EXPERIMENTS = 200
NOISE_LEVELS = [0.0, 0.1, 0.3]
SEED = 0
rng = np.random.default_rng(SEED)


def simulate(noise, n=N_ROWS):
    """Same shape as homework_6.1.csv, plus noise on Y. True effect = 1.2 + Z."""
    z = rng.uniform(0, 1, n)
    x = (rng.uniform(0, 1, n) < z).astype(int)   # higher Z -> more likely treated
    y = np.where(x == 1, 1.2 - 0.5 * z, -1.5 * z) + rng.normal(0, noise, n)
    return pd.DataFrame({"Z": z, "X": x, "Y": y})


def estimates(df):
    """(true optimal, max, 90th pct, smoothed max) for one dataset."""
    _, on_untreated = unit_effects(df)
    untreated = df[df["X"] == 0].assign(effect=on_untreated)

    true_optimal = 1.2 + untreated["Z"].max()
    line = smf.ols("effect ~ Z", untreated).fit()
    smoothed_max = line.predict(untreated).max()
    return true_optimal, on_untreated.max(), np.percentile(on_untreated, 90), smoothed_max


def report():
    print("homework_6.1.csv (no noise):")
    truth, raw_max, p90, smoothed = estimates(load())
    print(f"  true {truth:.4f}   max {raw_max:.4f}   90th pct {p90:.4f}   smoothed max {smoothed:.4f}")

    print(f"\nSimulated, {N_EXPERIMENTS} datasets per noise level -- mean error vs. true optimal (sd):")
    print(f"{'noise':>6} {'max':>16} {'90th pct':>16} {'smoothed max':>16}")
    for noise in NOISE_LEVELS:
        runs = np.array([estimates(simulate(noise)) for _ in range(N_EXPERIMENTS)])
        errors = runs[:, 1:] - runs[:, [0]]
        cells = [f"{m:+.3f} ({s:.3f})" for m, s in zip(errors.mean(0), errors.std(0, ddof=1))]
        print(f"{noise:6.1f} " + " ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    report()
