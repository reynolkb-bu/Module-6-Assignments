"""Week 11 -- event study t-test and placebo tests on simulated returns.

The target's return follows the market, plus a one-time +2 jump at the event.
We fit target ~ market on the days before the event, then divide the residual
on the day after by the training residuals' sd. That ratio is the t-value.

Answers: Q1 B, Q2 C, Q3 B, Q4 A.
"""

import numpy as np
import statsmodels.api as sm

NUM = 1000
EVENT = NUM // 2          # the real event_time from the quiz
JUMP_DAY = EVENT + 1      # the day the +2 jump actually happens


def make_error(corr_const, num, rng):
    """The quiz's correlated error: each value is mostly the last one."""
    sigma = 5 / np.sqrt((1 - corr_const) ** 2 / (1 - corr_const**2))
    prev = rng.normal(0, sigma)
    err = []
    for _ in range(num):
        prev = corr_const * prev + (1 - corr_const) * rng.normal(0, sigma)
        err.append(prev)
    return np.array(err)


def simulate(seed, corr_const=None):
    """The quiz's data. The same seed always gives the same dataset."""
    rng = np.random.default_rng(seed)
    R_market = rng.normal(0, 1, NUM) + np.arange(NUM) / NUM
    noise = rng.normal(0, 1, NUM) if corr_const is None else make_error(corr_const, NUM, rng)
    R_target = 2 + R_market + noise + (np.arange(NUM) == JUMP_DAY) * 2
    return R_market, R_target


def t_value(R_market, R_target, event_time):
    """Train on the days before event_time, test the day after it."""
    results = sm.OLS(R_target[:event_time], sm.add_constant(R_market[:event_time])).fit()
    resid = R_target - results.predict(sm.add_constant(R_market))
    return resid[event_time + 1] / resid[:event_time].std(ddof=2)


def placebo_rate(R_market, R_target):
    """Share of fake event times whose |t| > 1.96 (at least 3 training days)."""
    t = np.array([t_value(R_market, R_target, e) for e in range(3, NUM - 1)])
    return np.mean(np.abs(t) > 1.96)


def report():
    # Q1: t = (2 + noise) / 1, so it clears 1.96 about half the time.
    t = np.array([t_value(*simulate(seed), EVENT) for seed in range(2_000)])
    print(f"Q1 event detected in {np.mean(t > 1.96):.1%} of 2,000 datasets  -> B (0.5)")

    # Q2: one fixed dataset (seed 0), fake event at every possible time.
    print(f"Q2 placebo detection rate {placebo_rate(*simulate(0)):.3f}  -> C (0.05)")

    # Q3: 20 fake events before and 20 after the real one, on 300 datasets.
    fake_times = [*range(EVENT - 20, EVENT), *range(EVENT + 1, EVENT + 21)]
    shares = []
    for seed in range(300):
        R_market, R_target = simulate(seed)
        # Compare |t| -- a big negative placebo is as much a "fake event" as a big positive one.
        real = abs(t_value(R_market, R_target, EVENT))
        fake = np.abs([t_value(R_market, R_target, e) for e in fake_times])
        shares.append(np.mean(fake > real))
    print(f"Q3 share of placebos beating the real event {np.mean(shares):.3f}  -> B (0.15)")

    # Q4: same as Q2 but with correlated errors. The training sd already
    # absorbs the slow wander, so the false-positive rate stays near 5%.
    print(f"Q4 placebo detection rate, corr 0.9: {placebo_rate(*simulate(0, 0.9)):.3f}  -> A (0.05)")


if __name__ == "__main__":
    report()
