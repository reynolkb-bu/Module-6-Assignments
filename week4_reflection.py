"""Week 4 reflection -- the binned-W instrumental variable on homework_4.1.csv,
and college admission vs. test score near the cutoff (homework_4.2.a/b.csv).

Q1: split W into narrow ranges, take the Wald ratio in each, and average.
Equal-width ranges leave only a few rows (sometimes one Z group) at the tails,
so the ranges hold equal counts instead.

Q2: plot Y vs. X from 75 to 85. Y is 0/1, so the dots are the share admitted
in quarter-point bins with 95% intervals, against two logistic regressions:
a smooth one (Y ~ s) and one that can jump at 80 (Y ~ s + post + s:post).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

CSVS = Path(__file__).parent / "csvs"
PLOT = Path(__file__).parent / "week4_reflection.png"
W_BINS = 20
CUTOFF = 80
WINDOW = 5                 # plot 75 to 85
BIN_WIDTH = 0.25

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4df"
SMOOTH, JUMP = "#2a78d6", "#eb6834"


def load(name):
    return pd.read_csv(CSVS / name)


# --- Q1: instrumental variable, averaged over ranges of W ------------------

def wald(df):
    """(Y|Z=1 - Y|Z=0) / (X|Z=1 - X|Z=0)."""
    m = df.groupby("Z")[["X", "Y"]].mean()
    return (m.loc[1, "Y"] - m.loc[0, "Y"]) / (m.loc[1, "X"] - m.loc[0, "X"])


def ratios_by_w(df, bins):
    """Wald ratio in each range of W; ranges missing a Z group are skipped."""
    out = []
    for _, s in df.groupby(bins, observed=True):
        if s["Z"].nunique() == 2:
            out.append(wald(s))
    return np.array(out)


def report_instrument():
    df = load("homework_4.1.csv")
    print(f"overall ratio: {wald(df):.4f}\n")

    print(f"{'ranges':<16} {'mean':>7} {'min':>8} {'max':>8} {'skipped':>8}")
    for label, n, cut in [("equal count", 20, pd.qcut), ("equal count", 100, pd.qcut),
                          ("equal count", 250, pd.qcut), ("equal width", 10, pd.cut),
                          ("equal width", 20, pd.cut), ("equal width", 40, pd.cut)]:
        bins = cut(df["W"], n)
        r = ratios_by_w(df, bins)
        skipped = bins.value_counts().gt(0).sum() - len(r)
        print(f"{label + f' x{n}':<16} {r.mean():7.3f} {r.min():8.2f} {r.max():8.2f} {skipped:8d}")


# --- Q2: admission vs. test score near 80 ---------------------------------

def near_cutoff(name):
    df = load(name)
    df.columns = ["X", "Y"]
    df = df[df["X"].between(CUTOFF - WINDOW, CUTOFF + WINDOW)].copy()
    df["s"] = df["X"] - CUTOFF
    df["post"] = (df["X"] >= CUTOFF).astype(int)
    return df


def binned(df):
    """Share admitted per bin, with a 95% interval."""
    edges = np.arange(CUTOFF - WINDOW, CUTOFF + WINDOW + BIN_WIDTH, BIN_WIDTH)
    g = df.groupby(pd.cut(df["X"], edges, right=False), observed=True)["Y"]
    out = g.agg(["mean", "size"])
    out["x"] = [iv.mid for iv in out.index]
    out["ci"] = 1.96 * np.sqrt(out["mean"] * (1 - out["mean"]) / out["size"])
    return out


def draw(ax, df, title):
    smooth = smf.logit("Y ~ s", df).fit(disp=0)
    jump = smf.logit("Y ~ s * post", df).fit(disp=0)

    b = binned(df)
    ax.errorbar(b["x"], b["mean"], yerr=b["ci"], fmt="o", ms=4, color=MUTED,
                ecolor=MUTED, elinewidth=1, capsize=0, label="Share admitted (0.25-pt bins, 95% CI)")

    grid = np.linspace(-WINDOW, WINDOW, 400)
    curve = pd.DataFrame({"s": grid, "post": (grid >= 0).astype(int)})
    ax.plot(grid + CUTOFF, smooth.predict(curve), color=SMOOTH, lw=2, ls="--",
            label="Logistic, no cutoff term")
    for side in (grid < 0, grid >= 0):
        ax.plot(grid[side] + CUTOFF, jump.predict(curve[side]), color=JUMP, lw=2,
                label="Logistic with jump at 80" if side[0] else None)

    ax.axvline(CUTOFF, color=MUTED, lw=1, ls=":")
    ax.set_title(title, color=INK, loc="left")
    ax.set_xlabel("Test score (X)", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.spines[["top", "right"]].set_visible(False)
    return smooth, jump


def report_plot():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    print()
    for ax, name, title in [(axes[0], "homework_4.2.a.csv", "Dataset a"),
                            (axes[1], "homework_4.2.b.csv", "Dataset b")]:
        df = near_cutoff(name)
        smooth, jump = draw(ax, df, title)
        just_below = pd.DataFrame({"s": [-1e-9], "post": [0]})
        at_cutoff = pd.DataFrame({"s": [0.0], "post": [1]})
        print(f"{title}: n = {len(df)}   smooth at 80 = {smooth.predict(at_cutoff).iloc[0]:.3f}   "
              f"jump model {jump.predict(just_below).iloc[0]:.3f} -> "
              f"{jump.predict(at_cutoff).iloc[0]:.3f}   (post t = {jump.tvalues['post']:.1f})")

    axes[0].set_ylabel("Probability of getting into college (Y)", color=MUTED)
    axes[0].set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(PLOT, dpi=150)
    print(f"\nsaved {PLOT.name}")


if __name__ == "__main__":
    report_instrument()
    report_plot()
