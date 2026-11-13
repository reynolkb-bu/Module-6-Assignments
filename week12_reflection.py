"""Week 12 reflection -- DiD when prior trends don't hold.

Two groups, treatment for Group = 1 after Time = 0, true effect EFFECT. The
treated group also grows faster over the whole window (SLOPE_GAP per unit of
time), so the groups would have drifted apart even without treatment.

DiD = EFFECT + SLOPE_GAP * (mean post Time - mean pre Time)
    = 1 + 0.1 * (5 - (-5)) = 2
The extra trend gets counted as treatment effect, so DiD overstates it.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

rng = np.random.default_rng(0)

EFFECT = 1          # true treatment effect
SLOPE_GAP = 0.1     # how much faster the treated group grows per unit of time


def make_data(n=10_000):
    time = rng.uniform(-10, 10, n)
    group = (rng.uniform(size=n) < 0.5).astype(int)
    post = (time > 0).astype(int)
    y = (1 + 2 * group + 0.05 * time + SLOPE_GAP * group * time
         + EFFECT * group * post + rng.normal(0, 1, n))
    return pd.DataFrame({"Y": y, "Time": time, "Group": group, "post": post})


def report():
    df = make_data()

    pre = smf.ols("Y ~ Group * Time", df[df["Time"] < 0]).fit()
    print(f"Prior-trends test: Group:Time = {pre.params['Group:Time']:.3f},"
          f" t = {pre.tvalues['Group:Time']:.1f}")

    did = smf.ols("Y ~ Group * post", df).fit()
    time_gap = df.loc[df.post == 1, "Time"].mean() - df.loc[df.post == 0, "Time"].mean()
    print(f"True effect {EFFECT}")
    print(f"DiD estimate {did.params['Group:post']:.3f} (SE {did.bse['Group:post']:.3f})")
    print(f"Formula: {EFFECT} + {SLOPE_GAP} x {time_gap:.2f} = {EFFECT + SLOPE_GAP * time_gap:.3f}")


if __name__ == "__main__":
    report()
