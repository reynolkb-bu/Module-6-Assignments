"""Week 4 reflection -- the binned-W instrumental variable on homework_4.1.csv,
and college admission vs. test score near the cutoff (homework_4.2.a/b.csv).

Q1: split W into narrow ranges, take the Wald ratio in each, and average.
Equal-width ranges leave only a few rows (sometimes one Z group) at the tails,
so the ranges hold equal counts instead.

Q2: plot Y vs. X from 75 to 85. Y is 0/1, so the dots are the share admitted
in each half-point of score, against the probability predicted by a logistic
regression (Y ~ X). The curve is smooth, so it cannot follow the jump at 80.
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
BIN_WIDTH = 0.5

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4df"
DOTS, LINE = "#52514e", "#2a78d6"


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
    return df[df["X"].between(CUTOFF - WINDOW, CUTOFF + WINDOW)]


def draw(ax, df, title):
    """Dots: share admitted per half-point of score. Line: logistic regression."""
    edges = np.arange(CUTOFF - WINDOW, CUTOFF + WINDOW + BIN_WIDTH, BIN_WIDTH)
    share = df.groupby(pd.cut(df["X"], edges, right=False), observed=True)["Y"].mean()
    ax.scatter([iv.mid for iv in share.index], share * 100, s=22, color=DOTS,
               label="Actual % admitted")

    fit = smf.logit("Y ~ X", df).fit(disp=0)
    x = np.linspace(CUTOFF - WINDOW, CUTOFF + WINDOW, 200)
    ax.plot(x, fit.predict(pd.DataFrame({"X": x})) * 100, color=LINE, lw=2.5,
            label="Logistic regression prediction")

    # Actual share admitted in the one point of score on each side of 80.
    below = df.loc[df["X"].between(CUTOFF - 1, CUTOFF, inclusive="left"), "Y"].mean()
    above = df.loc[df["X"].between(CUTOFF, CUTOFF + 1, inclusive="left"), "Y"].mean()
    at_80 = fit.predict(pd.DataFrame({"X": [CUTOFF]})).iloc[0]
    ax.text(CUTOFF - WINDOW + 0.2, 5, f"Actual: jumps {below:.0%} to {above:.0%} at 80\n"
                             f"Logistic at 80: {at_80:.0%}", color=INK, fontsize=9)

    ax.axvline(CUTOFF, color=MUTED, lw=1, ls=":")
    ax.set_title(title, color=INK, loc="left")
    ax.set_xlabel("Test score", color=MUTED)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="y", color=GRID, lw=1)
    ax.spines[["top", "right"]].set_visible(False)
    return below, above, at_80


def report_plot():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    print()
    for ax, name, title in [(axes[0], "homework_4.2.a.csv", "Dataset a"),
                            (axes[1], "homework_4.2.b.csv", "Dataset b")]:
        below, above, at_80 = draw(ax, near_cutoff(name), title)
        print(f"{title}: actual {below:.0%} below 80, {above:.0%} above   logistic at 80 = {at_80:.0%}")

    axes[0].set_ylabel("Chance of getting into college", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(PLOT, dpi=150)
    print(f"\nsaved {PLOT.name}")


if __name__ == "__main__":
    report_instrument()
    report_plot()
