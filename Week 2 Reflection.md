# Week 2 Reflection

## 1. Invent an example situation that would use fixed effects.

An example situation that would use fixed effects would be a coffee chain. Let's say the chain has 30 different locations. They roll out a new menu and want to know whether the new menu accounted for an increase in sales week over week.

The issue with this is that the stores were never comparable. For example, a location at an airport would sell a lot more than a location at a mall in a town in the middle of nowhere.

Really, what you would want to do is compare the stores across locations, not the actual menu. Using fixed effects would fix this issue by giving each store its own baseline. Each store would get its own intercept and the shared menu would be estimated on top of that. The question then becomes: did each store beat its own weekly sales? 

## 2. Bootstrap the variance in the mean of a Pareto distribution. Explain what you had to do. As the sample size grows, what happens to that variance?

As the sample size grows, variance gets smaller. For example, think of Google reviews. A store with only a few Google reviews would be greatly impacted by one bad review or one good review. However, the more reviews a store gets, the less each individual review moves their average. This relates to Pareto values because the average becomes more stable. 

To bootstrap the variance of the mean of a Pareto distribution, I took one sample from the distribution. After that, I made 5,000 new samples by randomly picking values, allowed repeating values, and then took the average of each. The variance of the mean is how much those 5,000 averages vary.

```python
SHAPES = {"a = 3.0 (finite variance)": 3.0}
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
```

Each time the sample size increased by four times, the variance decreased to about 25% of what it was before. This essentially means that the variance will shrink at about 1/n. This in turn matches the Pareto theory. The one issue I found is that Pareto data sometimes will have big values, meaning a single run could seem scattered. Therefore I had to repeat each run 40 times and average those runs out.
