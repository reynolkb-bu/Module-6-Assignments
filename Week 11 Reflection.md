# Week 11 Reflection

## 1. Construct a dataset for an event study where the value, derivative, and second derivative of a trend all change discontinuously (suddenly) after an event. Build a model that tries to decide whether the event is real (has a nonzero effect) using (a) only the value, (b) the value, derivative, and second derivative. Which of these models is better at detecting and/or quantifying the impact of the event? (What might "better" mean here?)

A real-world example is a store that opens a new location. Sales at the new store jump immediately, which is the value. They start growing faster, which is the derivative. Overall growth speeds up as word gets out, the second derivative.

In my mock data, I use time as `s`, which is measured from the event at `s = 0`. Before the event, there is a straight and clear trend. After the event, the level increases by 2, the slope increases by 0.1, and the curvature increases by 0.002. 

```python
y = 10 + 0.05 * s + rng.normal(0, 1, n)
y += post * (2 + 0.1 * s + 0.002 * s**2)
```

I ended up fitting two models:

* (a) Value only: `y ~ s + post`, the event can only move the line up or down.
* (b) All three: `y ~ s + post + post:s + post:I(s**2)`, The event can impact the level, the slope, and the curvature.

I ran both models on 1,000 datasets which included the event and 1,000 datasets without the event.

| Model | Mean level estimate (true = 2) | Detected the real event | False positives (no event) |
|---|---|---|---|
| (a) Value only | -1.38 | 0% | 5.2% |
| (b) All three | 1.99 | 100% | 6.3% |

Model (b) ended up being better at both detecting that something happened and quantifying how big the effect is. Whereas model (a) tried to fit a rising curve with a flat step, which ended up tilting the line and got the increase backwards (-1.38 instead of +2). Furthermore, the curve also inflated the noise, which means that it never even detected the event. Both models had about 5% false positives, so the extra terms in model B don't cause extra false alarms. 

## 2. Construct a dataset in which there are three groups whose values each increase discontinuously (suddenly) by the same amount at a shared event; they change in parallel over time, but they have different starting values. Create a model that combines group fixed effects with an event study, as suggested in the online reading. Explain what you did, how the model works, and how it accounts for both baseline differences and the common event effect.

A good real-world example is to picture three stores in different cities that all increase their price on the same day. Since they sell different amounts to begin with but all grow at the same rate, they all get the same bump from the price change. 

I made three groups that had starting values of 0, 5, and 10. They shared a trend of 0.1 per period, and I had all of them jump by 3 at period 50. 

```python
y = base + 0.1 * t + 3 * (t >= 50) + rng.normal(0, 1, n)
```

The model is:

```python
smf.ols("y ~ C(group) + t + post", df).fit()
```

* `C(group)` Gives each group its own intercept. These are the group fixed effects, which absorb the baseline differences. 
* `t` is the trend that all groups share.
* `post` is 1 after the event. Its coefficient is the common event effect.

| Model | Estimated jump (true = 3) | Standard error |
|---|---|---|
| No group fixed effects | 3.31 | 0.98 |
| With group fixed effects | 3.31 | 0.23 |

The group effects came out to 0.07 for A, +5.13 for B, and +10.05 for C, which matches the true baselines of 0, 5, and 10.

I ended up comparing each group with its own past. By using fixed effects, we are able to take out where each group started, meaning that the only thing that's left is a shared trend and the jump that happens at the event itself. Because all three of the groups have the event at the same time, the model then uses all three to estimate the one jump. The estimate ends up being the same with or without fixed effects because the baselines aren't related to when the event happens. If we remove the fixed effects, the gaps between the groups would look like noise, and the standard error is about four times bigger. By adding in the fixed effects, it takes out that noise and makes the estimate a lot more accurate.
