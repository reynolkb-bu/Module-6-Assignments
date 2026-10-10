"""Week 1 reflection -- matching on homework_1.2.csv.

Q1: match each treated row (X = 1) to its nearest control (X = 0) on Z, then
look at how far the matches are and whether the far ones change the estimate.

Q2: weighted matching. Like (B), take every control within a radius of each
treated row, but weight each one by 1 - distance / radius so closer controls
count more. Each treated row's match is the weighted average Y of its controls.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

CSVS = Path(__file__).parent / "csvs"
PERCENTILES = [50, 90, 95, 100]      # median, 90th, 95th, farthest
CALIPERS = [np.inf, 0.05, 0.02]      # inf keeps every match
RADIUS = 0.2


def load():
    """Split homework_1.2.csv into treated (X = 1) and control (X = 0) rows."""
    df = pd.read_csv(CSVS / "homework_1.2.csv")
    return df, df[df["X"] == 1], df[df["X"] == 0]


# --- Q1: is the farthest match too far? -----------------------------------

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


def report_calipers():
    _, treated, control = load()
    distances, matched_y = nearest_matches(treated, control)
    treated_y = treated["Y"].to_numpy()

    # Drop matches farther than a caliper and see whether the effect moves.
    print(f"\n{'caliper':<9} {'matches kept':>13} {'effect':>8}")
    for caliper in CALIPERS:
        keep = distances <= caliper
        effect = (treated_y[keep] - matched_y[keep]).mean()
        print(f"{caliper:<9} {keep.sum():>13} {effect:8.3f}")


# --- Q2: weighted matching ------------------------------------------------

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


if __name__ == "__main__":
    report_distances()
    report_calipers()
    report_matching()
