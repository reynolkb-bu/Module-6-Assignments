"""Week 1 -- OLS on csvs/homework_1.1.csv, nearest-neighbor matching on csvs/homework_1.2.csv."""

from pathlib import Path

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


def load_data(path=DATA):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from {path.name}")
    print(df.head(), "\n")
    print(df.describe(), "\n")
    print("Missing values per column:")
    print(df.isna().sum(), "\n")
    print("Correlation matrix:")
    print(df.corr(), "\n")
    return df


def fit_model(df, features=FEATURES):
    X = sm.add_constant(df[features])
    y = df[TARGET]
    return sm.OLS(y, X).fit()


def main():
    df = load_data()
    model = fit_model(df)
    print(model.summary())

    # Q1: which value is the coefficient of X1 closest to? -> 1 (fitted 1.0071).
    coefs = model.params
    terms = " + ".join(f"{coefs[f]:.4f}*{f}" for f in FEATURES)
    print("\nFitted equation:")
    print(f"  Y = {coefs['const']:.4f} + {terms}")
    print(f"R-squared: {model.rsquared:.4f}   Adjusted R-squared: {model.rsquared_adj:.4f}")

    # Q2: which Xi differs most between its partial slope (all three Xi in the
    # model, so the others are held fixed) and its marginal slope (Xi alone)?
    # -> X2, because it correlates 0.90 with X1 and so absorbs X1's effect when
    # regressed on its own. X3 barely moves; it is uncorrelated with the others.
    print(f"\n{'var':<4} {'multiple':>10} {'simple':>10} {'|diff|':>10} {'t':>10}")
    for feature in FEATURES:
        simple = fit_model(df, [feature]).params[feature]
        gap = abs(coefs[feature] - simple)
        # Q3: most significant coefficient by t-statistic -> X3 (largest
        # coefficient, smallest standard error).
        print(f"{feature:<4} {coefs[feature]:10.4f} {simple:10.4f} {gap:10.4f} {model.tvalues[feature]:10.2f}")

    matching()


def matching(path=MATCH_DATA):
    """Match each X=1 row to its nearest X=0 row by Z (1-NN, with replacement)."""
    df = load_data(path)
    treated = df[df[MATCH_TREATMENT] == 1]
    control = df[df[MATCH_TREATMENT] == 0]

    nn = NearestNeighbors(n_neighbors=1).fit(control[[MATCH_COVARIATE]])
    distances, indices = nn.kneighbors(treated[[MATCH_COVARIATE]])
    matched = control.iloc[indices.ravel()]

    # Q4: distance of the farthest match -> 0.210217.
    print(f"Matched {len(treated)} X=1 rows to {indices.ravel().size} of "
          f"{len(control)} X=0 rows ({len(set(indices.ravel()))} distinct, matched with replacement)")
    print(f"Match distance: max {distances.max():.6f}, mean {distances.mean():.6f}")

    # Q5: effect = mean Y over all X=1 rows minus mean Y over the matched X=0
    # sample -> 0.54336, close to the 0.5 shift built into the data.
    effect = treated[TARGET].mean() - matched[TARGET].mean()
    print(f"Mean Y (X=1, full sample):     {treated[TARGET].mean():.6f}")
    print(f"Mean Y (X=0, matched sample):  {matched[TARGET].mean():.6f}")
    print(f"Effect:                        {effect:.6f}")


if __name__ == "__main__":
    main()
