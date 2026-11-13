# Week 12 Reflection

## Construct a dataset in which prior trends do not hold, and in which this makes the differences-in-differences come out wrong. Explain why the differences-in-differences estimate of the effect comes out higher or lower than the actual effect.

Differences-in-differences (DiD) shows how much the treated group changed before versus after. From there, you subtract how much a control group changed. This will only work if the two groups change by the same amount without the treatment. This assumption is called prior trends.

A real-world example is a company that launches a training program at its fastest-growing branch. Then it compares it to the training program at a slower branch. The fast branch was already growing at a lot quicker rate, so some of this growth wrongly gets credited to the training.

### The dataset

- There are two groups. Group 1 is treated after `Time = 0`.
- The true treatment effect is **1**.
- Group 1 also grows faster: **0.1 more per unit of time**, both before and after treatment. This breaks prior trends.

```python
y = (1 + 2 * group + 0.05 * time + 0.1 * group * time
     + 1 * group * post + rng.normal(0, 1, n))
```

### Results

| | Result |
|---|---|
| Prior-trends test (`Group:Time` before `Time = 0`) | 0.106, t = 10.8, so the trends are clearly not parallel |
| True effect | 1 |
| DiD estimate | 2.01 (SE 0.04) |

### Why DiD comes out too high

DiD ends up giving 2x the real effect. The extra 1 comes from:

1. The after-treatment period is about 10x later than the before-treatment period. 
2. During that time, Group 1 grows 0.1 x 10 = 1 more than Group 0, even without the treatment. 
3. DiD counts extra growth as part of the effect. 1 + 1 = 2.

The estimate ends up being too high because the treatment group was already going faster, per our real-world example. On the other hand, if it had been growing slower, then DiD would come out too low.
