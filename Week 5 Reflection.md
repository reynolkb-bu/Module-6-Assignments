# Week 5 Reflection

## 1. Draw a diagram for the following negative feedback loop: Sweating causes body temperature to decrease. High body temperature causes sweating.
## A negative feedback loop means that one thing increases another while the second thing decreases the first.
## Remember that we are using directed acyclic graphs where two things cannot directly cause each other.

```
Temp (1) ──(+)──> Sweat (1) ──(−)──> Temp (2) ──(+)──> Sweat (2) ──(−)──> Temp (3)
```

A directed acyclic graph can't have two things causing each other per the question. So I split up by time. For example, you're going on a run, one minute in you're hot and you start sweating. That's what cools you down. So at two minutes, you are a little bit cooler, so you start to sweat less. And then each moment causes the next one after that. 

## 2. Describe an example of a positive feedback loop. This means that one things increases another while the second things also increases the first.

An example of a positive feedback loop are Google reviews. A restaurant with a lot of five star reviews gets prioritized when people search. Therefore, a lot of people go there. The more customers that go there means the more reviews the restaurant will get. This brings in even more customers and puts the restaurant even higher up in the search priority. 

```
Reviews (week 1) ──(+)──> Customers (week 1) ──(+)──> Reviews (week 2) ──(+)──> Customers (week 2)
```

## 3. Draw a diagram for the following situation:

## Lightning storms frighten away deer and bears, decreasing their population, and cause flowers to grow, increasing their population.
## Bears eat deer, decreasing their population.
## Deer eat flowers, decreasing their population.

Write a dataset that simulates this situation.  (Show the code.) Include noise / randomness in all cases.

Identify a backdoor path with one or more confounders for the relationship between deer and flowers

```
                Lightning
          /         |           \
        (−)        (−)          (+)
       /            |             \
      v             v              v
    Bears ──(−)──> Deer ──(−)──> Flowers
```

Below is a code I used to simulate the lightning deer bear flower situation. The full code is in `week5_reflection.py`. Each row (n value) represents a season for a forest. Each deer eats 3 flowers per `3.0 * deer`. 

```python
rng = np.random.default_rng(0)
n = 10_000
lightning = np.maximum(rng.poisson(6, n) + rng.normal(0, 0.5, n), 0)
bears = np.maximum(20 - 1.0 * lightning + rng.normal(0, 3, n), 0)
deer = np.maximum(100 - 4.0 * lightning - 1.5 * bears + rng.normal(0, 8, n), 0)
flowers = np.maximum(600 + 25.0 * lightning - 3.0 * deer + rng.normal(0, 30, n), 0)
df = pd.DataFrame({"lightning": lightning, "bears": bears, "deer": deer, "flowers": flowers})

print(sm.OLS(df["flowers"], sm.add_constant(df[["deer"]])).fit().params["deer"])               # -6.18
print(sm.OLS(df["flowers"], sm.add_constant(df[["deer", "lightning"]])).fit().params["deer"])  # -3.04
```

The backdoor path that has one of our confounders for the relationship between deer and flowers is lightning. Deer <— Lightning —> Flowers. For example, a forest with high lightning has fewer deer and more flowers, but not due to the deer. The lightning is causing the fewer deer and more flowers. 

If you are only looking at the deer and flowers, it looks like each deer eats about 6 flowers. However, once you control for the lightning, it drops to 3, which is a real number. 

## 4. Draw a diagram for a situation of your own invention.  The diagram should include at least four nodes, one confounder, and one collider.  Be sure that it is acyclic (no loops).  Which node would say is most like a treatment (X)?  Which is most like an outcome (Y)?

```
                 Motivation
                /          \
             (+)            (+)
              /              \
             v                v
    Going to the gym     Eating better
       |        \             /
      (+)       (+)         (+)
       |          \         /
       |           v       v
       |         Losing weight
       |               |
       |              (+)
       v               |
   Buying new clothes <┘
```

- **Treatment (X):** going to the gym.
- **Outcome (Y):** losing weight.
- **Confounder:** motivation. A person who is motivated goes to the gym more. However, a person that is motivated also eats better. Eating better alone helps them lose weight, regardless of if they go to the gym or not.
- **Collider:** buying new clothes. People who go to the gym buy workout clothes. Similarly, people who lose weight need to buy smaller clothes because their current clothes are too big.

There are two paths from the gym to losing weight:

- **Direct path:** Gym → Losing weight. This is what we actually want to measure.
- **Backdoor path:** Gym ← Motivation → Eating better → Losing weight. When we control for motivation, this path closes, leaving only the direct path, which is what we want.

We should control for motivation, but do not care about controlling for buying new clothes. The reason is, if we only looked at people who bought new clothes, we would be mixing up the gym goers and people who lost weight another way. For example, taking weight loss drugs. This would lead to the gym looking like it helps less than it actually does.
