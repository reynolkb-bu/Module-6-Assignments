"""Week 8 -- inverse probability weighting (homework_8.1.csv) and Mahalanobis
matching (homework_8.2.csv).

Answers: Q1 B, Q2 A, Q3 D, Q4 D.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.linear_model import LogisticRegression

CSVS = Path(__file__).parent / "csvs"
DATA_IPW = CSVS / "homework_8.1.csv"
DATA_MATCH = CSVS / "homework_8.2.csv"


def load(path):
    return pd.read_csv(path)


# --- Q1-Q2: inverse probability weighting -----------------------------------

def propensity_scores(df):
    """P(X = 1 | Z) from a logistic regression of X on Z."""
    model = LogisticRegression().fit(df[["Z"]], df["X"])
    # predict_proba gives one column per class, [P(X = 0), P(X = 1)].
    return model.predict_proba(df[["Z"]])[:, 1]


def ipw_ate(df, p):
    """Weighted mean of Y for treated minus weighted mean of Y for untreated."""
    treated = df["X"] == 1
    weights = np.where(treated, 1 / p, 1 / (1 - p))
    y_treated = np.average(df["Y"][treated], weights=weights[treated])
    y_untreated = np.average(df["Y"][~treated], weights=weights[~treated])
    return y_treated - y_untreated


def report_ipw():
    df = load(DATA_IPW)
    p = propensity_scores(df)
    print(f"Q1 IPW ATE: {ipw_ate(df, p):.4f}  -> B (2.3)")
    print(f"Q2 first three propensity scores: {np.round(p[:3], 2)}  -> A [0.84, 0.58, 0.71]")


# --- Q3-Q4: Mahalanobis matching --------------------------------------------

def inverse_covariance(df):
    """Invert the 2x2 covariance of all Z1 and Z2 values (a 2xN matrix)."""
    z = df[["Z1", "Z2"]].to_numpy().T
    return np.linalg.inv(np.cov(z))


def match_treated(df):
    """For each treated row: its Y minus its nearest untreated Y, the distance to
    that match, and the match's row.

    With replacement -- every treated row searches all untreated rows, so the
    same untreated row can be the match more than once.
    """
    vi = inverse_covariance(df)
    treated = df[df["X"] == 1]
    untreated = df[df["X"] == 0]
    untreated_z = untreated[["Z1", "Z2"]].to_numpy()

    effects, distances, matches = [], [], []
    for z, y in zip(treated[["Z1", "Z2"]].to_numpy(), treated["Y"]):
        d = np.array([mahalanobis(z, u, vi) for u in untreated_z])
        nearest = d.argmin()
        effects.append(y - untreated["Y"].iloc[nearest])
        distances.append(d[nearest])
        matches.append(untreated.iloc[nearest])
    return treated, np.array(effects), np.array(distances), matches


def report_matching():
    df = load(DATA_MATCH)
    treated, effects, distances, matches = match_treated(df)
    print(f"\nQ3 Mahalanobis-matched ATE: {effects.mean():.4f}  -> D (3.4)")

    # Least common support = the treated row whose best match is still the farthest away.
    # Q4 asks for that row's nearest match, not the treated row itself.
    worst = distances.argmax()
    row, match = treated.iloc[worst], matches[worst]
    print(f"Q4 least common support: treated row at Z1 = {row['Z1']:.4f}, Z2 = {row['Z2']:.4f}")
    print(f"   its nearest untreated match: Z1 = {match['Z1']:.4f}, Z2 = {match['Z2']:.4f}, "
          f"distance {distances[worst]:.4f}  -> D (1.5, -1.3)")


def report():
    report_ipw()
    report_matching()


if __name__ == "__main__":
    report()
