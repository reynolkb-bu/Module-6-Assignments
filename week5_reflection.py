"""Week 5 reflection Q3 -- lightning, bears, deer, and flowers.

    lightning -> bears    (-)  storms frighten bears away
    lightning -> deer     (-)  storms frighten deer away
    lightning -> flowers  (+)  storms make flowers grow
    bears     -> deer     (-)  bears eat deer
    deer      -> flowers  (-)  deer eat flowers

One row is one forest region in one season. Every node gets its own noise.

Backdoor paths from deer to flowers:
    deer <- lightning -> flowers
    deer <- bears <- lightning -> flowers
Lightning is on both, so controlling for lightning closes them. Bears is not
needed once lightning is in: its only way to flowers besides deer is lightning.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

N_REGIONS = 10_000
DEER_ON_FLOWERS = -3.0    # true effect: each deer eats 3 flowers
SEED = 0


def simulate(n=N_REGIONS, seed=SEED):
    rng = np.random.default_rng(seed)
    lightning = np.maximum(rng.poisson(6, n) + rng.normal(0, 0.5, n), 0)   # storms per season
    bears = np.maximum(20 - 1.0 * lightning + rng.normal(0, 3, n), 0)
    deer = np.maximum(100 - 4.0 * lightning - 1.5 * bears + rng.normal(0, 8, n), 0)
    flowers = np.maximum(
        600 + 25.0 * lightning + DEER_ON_FLOWERS * deer + rng.normal(0, 30, n), 0)
    return pd.DataFrame({"lightning": lightning, "bears": bears,
                         "deer": deer, "flowers": flowers})


def coef_of_deer(df, controls=()):
    X = sm.add_constant(df[["deer", *controls]])
    return sm.OLS(df["flowers"], X).fit().params["deer"]


def report():
    df = simulate()
    print(df.describe().loc[["mean", "std", "min", "max"]].round(2))
    print("\ncorrelations:")
    print(df.corr().round(3))

    print(f"\n{'flowers ~':<30} {'coef of deer':>13}")
    for controls in [(), ("bears",), ("lightning",), ("lightning", "bears")]:
        label = " + ".join(["deer", *controls])
        print(f"{label:<30} {coef_of_deer(df, controls):13.4f}")
    print(f"{'true effect':<30} {DEER_ON_FLOWERS:13.4f}")


if __name__ == "__main__":
    report()
