# Week 1 Reflection

## 1. In Coding Quiz 1, you are asked to find the distance of the farthest match in a set. Is this farthest match distance too far to be a meaningful match? How can you decide this?

Yes, a distance of the farthest match in a set is too great to be a meaningful match. The reason is that matching usually only works if the pair is similar. For example, pricing a kitchen remodel by comparing a remodeled house to one that has not been remodeled. This would be like comparing a house that is twice as large. The pricing difference would be about the size of the house, not the actual kitchen.

One way to decide is to compare the gap in the data spread. The furthest match is about 0.21 apart, which is about 0.73 standard deviations of Z. A common rule of thumb we can use allows for about 0.2 standard deviations. The typical match here is only 0.013 apart. This is due to 18 of the 48 treated rows having a larger Z than every control, so they all get matched to the same edge control.

```python
CSVS = Path(__file__).parent / "csvs"
PERCENTILES = [50, 90, 95, 100]      # median, 90th, 95th, farthest

def load():
    """Split homework_1.2.csv into treated (X = 1) and control (X = 0) rows."""
    df = pd.read_csv(CSVS / "homework_1.2.csv")
    return df, df[df["X"] == 1], df[df["X"] == 0]

def nearest_matches(treated, control):
    """Approach A: nearest control on Z, with replacement. Returns distances and matched Y."""
    nn = NearestNeighbors(n_neighbors=1).fit(control[["Z"]])
    distances, indices = nn.kneighbors(treated[["Z"]])
    return distances.ravel(), control["Y"].to_numpy()[indices.ravel()]

def report_distances():
    df, treated, control = load()
    distances, _ = nearest_matches(treated, control)

    print(f"farthest match distance:     {distances.max():.4f}")
    print(f"median match distance:       {np.median(distances):.4f}")
    print(f"farthest, in SDs of Z:       {distances.max() / df['Z'].std():.2f}  (rule of thumb: 0.2)")
    print(f"treated above every control: {(treated['Z'] > control['Z'].max()).sum()} of {len(treated)}")

    # The farthest match against the rest: is it typical or an outlier?
    print(f"\n{'percentile':<11} {'distance':>9}")
    for p, value in zip(PERCENTILES, np.percentile(distances, PERCENTILES)):
        print(f"{p:<11} {value:9.4f}")
```

The way we can be certain about this is to check whether the far matches actually change the answer. If we use all the matches, the effect is 0.543. If we keep only the matches within 0.05, it gives us 0.503. If we keep all the matches within 0.02, it gives us 0.500. The estimate improves once the bad matches are dropped, which proves that they were indeed the problem.

```python
CALIPERS = [np.inf, 0.05, 0.02]      # inf keeps every match

def report_calipers():
    df, treated, control = load()
    distances, matched_y = nearest_matches(treated, control)
    treated_y = treated["Y"].to_numpy()

    # Y - Z removes Z's part of Y, so the gap between the groups is the true treatment effect.
    true_effect = (df["Y"] - df["Z"]).groupby(df["X"]).mean().diff().iloc[-1]
    print(f"\ntrue effect: {true_effect:.3f}")

    # Drop matches farther than a caliper and see whether the effect moves.
    print(f"\n{'caliper':<9} {'matches kept':>13} {'effect':>8}")
    for caliper in CALIPERS:
        keep = distances <= caliper
        effect = (treated_y[keep] - matched_y[keep]).mean()
        print(f"{caliper:<9} {keep.sum():>13} {effect:8.3f}")
```

## 2. Invent your own type of matching similar to (A) and (B), which has a different way to pick the matches in X = 0.

The approach I would use that is similar to A and B would be weighted matching. Similar to approach B, I would take every control within a set of each treated row. The difference is that closer controls count more than the farther ones. A control right next to the treated row would be weighted more, whereas a control near the edge would not be weighted as much.

When we use approach A, it picks the control that is the closest, regardless of other controls being almost as close as the one that gets picked. When using approach B, it will look at every nearby control. Let's say there is one at 0.0 and another at 0.9. It would weight these equally, which doesn't really make sense.

Weighted matching is a balanced way of doing it. It would be like pricing a house where you look at houses that are nearby but put more weight on the ones that are very similar to each other.

```python
RADIUS = 0.05

def radius_matching(treated, control, radius=RADIUS, weighted=True):
    """Each treated row vs. its controls within `radius`. weighted=False is approach B."""
    control_y = control["Y"].to_numpy()
    nn = NearestNeighbors(radius=radius).fit(control[["Z"]])
    dists, groups = nn.radius_neighbors(treated[["Z"]])
    effects = []
    for y, d, group in zip(treated["Y"].to_numpy(), dists, groups):
        if len(group):
            weights = 1 - d / radius if weighted else None
            effects.append(y - np.average(control_y[group], weights=weights))
    return np.mean(effects)

def report_matching():
    _, treated, control = load()
    _, matched_y = nearest_matches(treated, control)

    print(f"\n(A) nearest neighbor:        {(treated['Y'].to_numpy() - matched_y).mean():.3f}")
    print(f"(B) radius, equal weights:   {radius_matching(treated, control, weighted=False):.3f}")
    print(f"(C) radius, closer weighted: {radius_matching(treated, control, weighted=True):.3f}")
```
