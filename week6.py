"""Week 6 -- treatment effects from nearest-neighbor matching on csvs/homework_6.1.csv.

Each row is matched on the confounder Z to its nearest row in the other group
(X = 1 treated, X = 0 untreated). That match is the row's counterfactual.

Answers: Q1 D, Q2 B, Q3 B, Q4 C.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

CSVS = Path(__file__).parent / "csvs"
DATA = CSVS / "homework_6.1.csv"


def load(path=DATA):
    return pd.read_csv(path)


def counterfactual_y(rows, other_group):
    """Y of the nearest row (by Z) in the other group, one per row."""
    nn = NearestNeighbors(n_neighbors=1).fit(other_group[["Z"]])
    _, indices = nn.kneighbors(rows[["Z"]])
    return other_group["Y"].to_numpy()[indices.ravel()]


def unit_effects(df):
    """Treated minus untreated for every row: (effects on treated, effects on untreated)."""
    treated = df[df["X"] == 1]
    untreated = df[df["X"] == 0]
    # Treated row: its own Y is the treated outcome, the match is what it would be untreated.
    on_treated = treated["Y"].to_numpy() - counterfactual_y(treated, untreated)
    # Untreated row: the match is what it would be treated, its own Y is the untreated outcome.
    on_untreated = counterfactual_y(untreated, treated) - untreated["Y"].to_numpy()
    return on_treated, on_untreated


def report():
    df = load()
    on_treated, on_untreated = unit_effects(df)

    ate = np.concatenate([on_treated, on_untreated]).mean()
    att = on_treated.mean()
    atu = on_untreated.mean()
    optimal = on_untreated.max()

    print(f"{len(on_treated)} treated, {len(on_untreated)} untreated")

    # Q1: all 1,000 rows, each with its own counterfactual -> 1.6953.
    # Q2: only the 491 treated rows -> 1.8464.
    # Q3: only the 509 untreated rows -> 1.5495.
    # ATT is above ATU because the effect grows with Z (about 1.2 + Z), and the
    # treated rows sit at much higher Z (mean 0.65 vs 0.35 for untreated).
    # Q4: the largest single untreated effect -> 2.1725 (the untreated row with Z = 0.973).
    print(f"\nQ1 average treatment effect (ATE):     {ate:.4f}  -> D (1.695)")
    print(f"Q2 effect on the treated (ATT):        {att:.4f}  -> B (1.846)")
    print(f"Q3 effect on the untreated (ATU):      {atu:.4f}  -> B (1.549)")
    print(f"Q4 optimal effect (max untreated):     {optimal:.4f}  -> C (2.172)")


if __name__ == "__main__":
    report()
