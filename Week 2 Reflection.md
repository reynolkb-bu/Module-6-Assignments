# Week 2 Reflection

## 1. Invent an example situation that would use fixed effects.

An example situation that would use fixed effects would be a coffee chain. Let's say the chain has 30 different locations. They roll out a new menu and want to know whether the new menu accounted for an increase in sales week over week.

The issue with this is that the stores were never comparable. For example, a location at an airport would sell a lot more than a location at a mall in a town in the middle of nowhere.

Really, what you would want to do is compare the stores across locations, not the actual menu. Using fixed effects would fix this issue by giving each store its own baseline. Each store would get its own intercept and the shared menu would be estimated on top of that. The question then becomes: did each store beat its own weekly sales? 

## 2. Bootstrap the variance in the mean of a Pareto distribution. Explain what you had to do. As the sample size grows, what happens to that variance?

As the sample size grows, variance gets smaller. For example, think of Google reviews. A store with only a few Google reviews would be greatly impacted by one bad review or one good review. However, the more reviews a store gets, the less each individual review moves their average. This relates to Pareto values because the average becomes more stable. 

To bootstrap the variance of the mean of a Pareto distribution, I took one sample from the distribution. After that, I made 5,000 new samples by randomly picking values, allowed repeating values, and then took the average of each. The variance of the mean is how much those 5,000 averages vary.

```python
def pareto_sample(rng, n, a):
    return 1.0 + rng.pareto(a, n)  # numpy's pareto() starts at 0, so shift it to start at 1

def bootstrap_mean_variance(sample, rng):
    idx = rng.integers(0, len(sample), size=(5_000, len(sample)))  # 5,000 resamples, with repeats
    return sample[idx].mean(axis=1).var(ddof=1)                    # variance of the 5,000 means

rng = np.random.default_rng(0)
for n in [100, 400, 1600, 6400, 25600]:
    print(n, np.mean([bootstrap_mean_variance(pareto_sample(rng, n, 3.0), rng) for _ in range(40)]))
```

Every time the sample size increased by 4x, the variance dropped to about 25% of what it was before, which makes sense. This means the variance shrinks at about 1/n, which matches the Pareto theory. The one issue I found is that Pareto data sometimes will have big values, meaning a single run could seem scattered. Therefore I had to repeat each run 40 times and average those runs out.
