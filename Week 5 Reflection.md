# Week 5 Reflection

## 1. Draw a diagram for the following negative feedback loop: Sweating causes body temperature to decrease. High body temperature causes sweating.

```
Temp (1:00) ──(+)──> Sweat (1:00) ──(−)──> Temp (1:05) ──(+)──> Sweat (1:05) ──(−)──> Temp (1:10)
```

A DAG can't have two things causing each other, so I split it up by time. Think of going for a run. At 1:00 you're hot, so you start sweating. The sweat cools you down, so at 1:05 you're a little cooler, so you sweat a little less. Each moment only causes the next one, so there's no loop.

## 2. Describe an example of a positive feedback loop.

Google reviews. A restaurant with a lot of good reviews shows up first when people search, so more people go there. More customers means more reviews, which brings in even more customers.

```
Reviews (week 1) ──(+)──> Customers (week 1) ──(+)──> Reviews (week 2) ──(+)──> Customers (week 2)
```

## 3. Draw a diagram for the lightning, bears, deer, and flowers situation. Write a dataset that simulates it, and identify a backdoor path for the relationship between deer and flowers.

```
                Lightning
             /      |      \
          (−)      (−)      (+)
           /        |        \
          v         v         v
      Bears ──(−)──> Deer ──(−)──> Flowers
```

This is the code I used to simulate it (full version in `week5_reflection.py`). Each row is one forest for one season, and each deer eats 3 flowers.

```python
rng = np.random.default_rng(0)
n = 10_000
lightning = np.maximum(rng.poisson(6, n) + rng.normal(0, 0.5, n), 0)
bears = np.maximum(20 - 1.0 * lightning + rng.normal(0, 3, n), 0)
deer = np.maximum(100 - 4.0 * lightning - 1.5 * bears + rng.normal(0, 8, n), 0)
flowers = np.maximum(600 + 25.0 * lightning - 3.0 * deer + rng.normal(0, 30, n), 0)
df = pd.DataFrame({"lightning": lightning, "bears": bears, "deer": deer, "flowers": flowers})
```

The backdoor path is Deer ← Lightning → Flowers, and lightning is the confounder. A stormy forest has fewer deer and more flowers, but not because of the deer. The storms are doing both.

If I just compare deer and flowers, it looks like each deer eats about 6 flowers. Once I control for lightning, it drops to 3, which is the real answer.

## 4. Draw a diagram for a situation of your own invention. Which node is most like a treatment (X)? Which is most like an outcome (Y)?

```
             Motivation
             /        \
          (+)          (+)
           /            \
          v              v
   Going to the gym ──(+)──> Losing weight
           \              /
          (+)          (+)
             \        /
              v      v
          Buying new clothes
```

- **Treatment (X):** going to the gym.
- **Outcome (Y):** losing weight.
- **Confounder:** motivation. Someone who is motivated goes to the gym more, but they also eat better, which helps them lose weight on its own.
- **Collider:** buying new clothes. People who go to the gym buy workout clothes, and people who lose weight buy smaller clothes.

We should control for motivation, but not for buying new clothes. If we only looked at people who bought new clothes, we would be mixing gym-goers with people who lost weight some other way, and the gym would look like it helps less than it actually does.
