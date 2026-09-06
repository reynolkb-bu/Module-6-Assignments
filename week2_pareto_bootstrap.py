"""Week 2 reflection Q2 -- how does var(sample mean) change as the sample grows?

Draw a sample of size n from a Pareto, resample it with replacement B times, and
take the variance of those B means. Repeat as n grows to read off the rate.

Pareto(shape a, scale 1) has a mean only when a > 1 and a finite variance only
when a > 2, so a = 3.0 converges at the usual 1/n rate and a = 1.5 does not.
"""

import numpy as np

SHAPES = {"a = 3.0 (finite variance)": 3.0, "a = 1.5 (infinite variance)": 1.5}
SIZES = [100, 400, 1600, 6400, 25600]  # quadruples, so 1/n predicts a ratio of 0.25
N_BOOTSTRAP = 5_000
N_REPLICATES = 40  # one sample alone is far too noisy on a heavy tail
SEED = 0


def pareto_sample(rng, n, a):
    """numpy's pareto() is Lomax and starts at 0, so shift it to start at 1."""
    return 1.0 + rng.pareto(a, n)


def bootstrap_mean_variance(sample, rng):
    """Resample with replacement B times, return the variance of the B means."""
    idx = rng.integers(0, len(sample), size=(N_BOOTSTRAP, len(sample)))
    return sample[idx].mean(axis=1).var(ddof=1)


def theoretical(a, n):
    """var(X)/n, which only exists for a > 2."""
    return a / ((a - 1) ** 2 * (a - 2)) / n if a > 2 else float("nan")


def run(label, a):
    rng = np.random.default_rng(SEED)
    print(f"\n{label}")
    print(f"{'n':>7} {'bootstrap var(mean)':>21} {'theory var(X)/n':>17} {'ratio vs prev':>14}")

    previous = None
    for n in SIZES:
        variance = float(np.mean([
            bootstrap_mean_variance(pareto_sample(rng, n, a), rng)
            for _ in range(N_REPLICATES)
        ]))
        ratio = "-" if previous is None else f"{variance / previous:.3f}"
        print(f"{n:>7} {variance:21.6f} {theoretical(a, n):17.6f} {ratio:>14}")
        previous = variance


if __name__ == "__main__":
    for label, shape in SHAPES.items():
        run(label, shape)
    print("\nSample size quadruples each row, so 1/n predicts a ratio near 0.25.")
