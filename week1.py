"""Week 1 -- OLS on csvs/homework_1.1.csv, nearest-neighbor matching on csvs/homework_1.2.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors

CSVS = Path(__file__).parent / "csvs"
DATA = CSVS / "homework_1.1.csv"
MATCH_DATA = CSVS / "homework_1.2.csv"
FEATURES = ["X1", "X2", "X3"]
TARGET = "Y"
MATCH_TREATMENT = "X"
MATCH_COVARIATE = "Z"
MATCH_RADIUS = 0.2


def load_data(path=DATA):
    return pd.read_csv(path)


def fit_model(df, features=FEATURES):
    X = sm.add_constant(df[features])
    y = df[TARGET]
    return sm.OLS(y, X).fit()


def regression():
    df = load_data()
    model = fit_model(df)
    coefs = model.params

    simple = {f: fit_model(df, [f]).params[f] for f in FEATURES}
    gaps = {f: abs(coefs[f] - simple[f]) for f in FEATURES}

    print(f"{'var':<4} {'multiple regression':>10} {'simple regression':>10} {'|diff|':>10} {'t':>10}")
    for f in FEATURES:
        print(f"{f:<4} {coefs[f]:10.4f} {simple[f]:10.4f} {gaps[f]:10.4f} {model.tvalues[f]:10.2f}")

    # Q1: coefficient of X1 -> 1.0071, closest to 1 (B).
    # Q2: largest |multiple - simple| -> X2 (A); it correlates 0.90 with X1 and so
    #     absorbs X1's effect when regressed on its own. X3 barely moves.
    # Q3: largest |t| -> X3 (C).
    print(f"\nQ1 coefficient of X1:            {coefs['X1']:.4f}  -> B (1)")
    print(f"Q2 greatest multiple regression/simple regression gap: {max(gaps, key=gaps.get)}  -> A")
    print(f"Q3 largest |t-statistic|:        {model.tvalues[FEATURES].abs().idxmax()}  -> C")


def matching(path=MATCH_DATA):
    """Q4-Q5: match each X=1 row to its nearest X=0 row by Z (1-NN, with replacement)."""
    df = load_data(path)
    treated = df[df[MATCH_TREATMENT] == 1]
    control = df[df[MATCH_TREATMENT] == 0]

    nn = NearestNeighbors(n_neighbors=1).fit(control[[MATCH_COVARIATE]])
    distances, indices = nn.kneighbors(treated[[MATCH_COVARIATE]])
    matched = control.iloc[indices.ravel()]

    # Q5: effect = mean Y over all X=1 rows minus mean Y over the matched X=0
    # sample -> 0.54336, close to the 0.5 shift built into the data.
    effect = treated[TARGET].mean() - matched[TARGET].mean()
    print(f"\nQ4 farthest match distance:   {distances.max():.6f}  -> A")
    print(f"Q5 effect (1-NN matching):    {effect:.6f}  -> A")


def matching_radius(path=MATCH_DATA, radius=MATCH_RADIUS):
    """Q6-Q7: approach B, match each X=1 row to every X=0 row within `radius` on Z."""
    df = load_data(path)
    treated = df[df[MATCH_TREATMENT] == 1]
    control = df[df[MATCH_TREATMENT] == 0]

    nn = NearestNeighbors(radius=radius).fit(control[[MATCH_COVARIATE]])
    _, indices = nn.radius_neighbors(treated[[MATCH_COVARIATE]])
    matched = np.concatenate(indices)

    # Q6: duplicates = every appearance of an X=0 row after its first -> 685.
    duplicates = matched.size - len(set(matched))

    # Q7: average the Y of each neighbor group first, then average those group
    # means, so a big group does not outweigh a small one -> effect 0.584412.
    group_means = np.array([control[TARGET].iloc[group].mean() for group in indices if len(group)])
    effect = treated[TARGET].mean() - group_means.mean()
    print(f"\nQ6 duplicates (all but first): {duplicates}  -> C")
    print(f"Q7 effect (radius matching):  {effect:.6f}  -> A")


def main():
    regression()
    matching()
    matching_radius()


if __name__ == "__main__":
    main()
