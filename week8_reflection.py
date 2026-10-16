"""Week 8 reflection -- checks on the choices made in week8.py.

IPW   naive difference vs. IPW, two ways to average the weights, and sklearn's
      default regularization vs. none.
Match how often untreated rows get reused, and Mahalanobis vs. plain Euclidean.
"""

import numpy as np
from scipy.spatial.distance import euclidean, mahalanobis
from sklearn.linear_model import LogisticRegression

from week8 import (DATA_IPW, DATA_MATCH, inverse_covariance, ipw_ate, load,
                   propensity_scores)


def ipw_divide_by_n(df, p):
    """Same weights, but each group's weighted sum is divided by N, not by the weights."""
    treated = df["X"] == 1
    n = len(df)
    return (df["Y"][treated] / p[treated]).sum() / n - (df["Y"][~treated] / (1 - p[~treated])).sum() / n


def nearest_untreated(df, distance):
    """Index of the nearest untreated row and the effect, for every treated row."""
    treated = df[df["X"] == 1]
    untreated = df[df["X"] == 0]
    untreated_z = untreated[["Z1", "Z2"]].to_numpy()
    matches, effects = [], []
    for z, y in zip(treated[["Z1", "Z2"]].to_numpy(), treated["Y"]):
        nearest = np.array([distance(z, u) for u in untreated_z]).argmin()
        matches.append(untreated.index[nearest])
        effects.append(y - untreated["Y"].iloc[nearest])
    return matches, np.array(effects)


def report_ipw():
    df = load(DATA_IPW)
    p = propensity_scores(df)
    naive = df[df["X"] == 1]["Y"].mean() - df[df["X"] == 0]["Y"].mean()
    p_unpenalized = LogisticRegression(C=np.inf).fit(df[["Z"]], df["X"]).predict_proba(df[["Z"]])[:, 1]

    print("homework_8.1.csv")
    print(f"  naive difference in means:      {naive:.4f}")
    print(f"  IPW, weighted means (week8.py): {ipw_ate(df, p):.4f}")
    print(f"  IPW, sums divided by N:         {ipw_divide_by_n(df, p):.4f}")
    print(f"  IPW, no regularization:         {ipw_ate(df, p_unpenalized):.4f}")
    weights = np.where(df["X"] == 1, 1 / p, 1 / (1 - p))
    print(f"  propensity range {p.min():.3f} to {p.max():.3f}, largest weight {weights.max():.2f}")


def report_matching():
    df = load(DATA_MATCH)
    vi = inverse_covariance(df)
    naive = df[df["X"] == 1]["Y"].mean() - df[df["X"] == 0]["Y"].mean()
    m_matches, m_effects = nearest_untreated(df, lambda a, b: mahalanobis(a, b, vi))
    _, e_effects = nearest_untreated(df, euclidean)

    print("\nhomework_8.2.csv")
    print(f"  naive difference in means: {naive:.4f}")
    print(f"  Mahalanobis matching:      {m_effects.mean():.4f}")
    print(f"  Euclidean matching:        {e_effects.mean():.4f}")
    print(f"  sd of Z1 {df['Z1'].std():.3f}, sd of Z2 {df['Z2'].std():.3f}")
    print(f"  {len(m_matches)} treated rows matched to {len(set(m_matches))} distinct untreated rows "
          f"(of {(df['X'] == 0).sum()})")


if __name__ == "__main__":
    report_ipw()
    report_matching()
