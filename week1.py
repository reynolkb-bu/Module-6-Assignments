"""Week 1 -- linear regression predicting Y from X1, X2, X3 (csvs/homework_1.1.csv)."""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

DATA = Path(__file__).parent / "csvs" / "homework_1.1.csv"
FEATURES = ["X1", "X2", "X3"]
TARGET = "Y"


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


if __name__ == "__main__":
    main()
