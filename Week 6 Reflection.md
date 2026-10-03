# Week 6 Reflection

## 1. What is a potential problem with computing the Marginal Treatment Effect simply by comparing each untreated item to its counterfactual and taking the maximum difference?

A potential problem with computing the marginal treatment effect by comparing the untreated item to its counterfactual and taking the maximum difference is that the item picked is probably due to luck. Each item's Effect is a combination of its real effect with some noise added. When we take the max, we pick the item where the noise happened to be the highest, since we only are using one item. We don't average that noise out. 

For example, imagine having a room full of people that flip a coin ten times. Someone will probably get nine or ten heads, but that doesn't mean their coin is better. It simply means they got lucky. With 509 untreated items, we have 509 chances for one of them to get lucky. 

The quiz has no noise, so the max of 2.172 is correct. I simulated the same data with noise added to y, and the max overshot the true answer by 0.20 with little noise and 0.85 with more noise. It never came in low, meaning it's not a random error. It just happens to be too high.

## 2. Propose a solution that remedies this problem and write some code that implements your solution. It's very important here that you clearly explain what your solution will do.

My solution. I propose that remedies this problem is smooth max. Instead of us trusting the item with the biggest effect, I'll use all 509 treated items, and draw a line of how that effect changes with Z. From there, I will take the highest point on the line, because one lucky item will barely move from a line that's built from 509 items. 

A good real-world example that this relates to would be a school that wants to know who tutoring helps most. The student whose grade jumped the highest might have had a good test day. Instead, you need to look at all the students. If students who started with lower grades improved more overall, then that is a trend we can trust more than just looking at one student.

```python
_, on_untreated = unit_effects(df)
untreated = df[df["X"] == 0].assign(effect=on_untreated)

line = smf.ols("effect ~ Z", untreated).fit()
smoothed_max = line.predict(untreated).max()
```

I ran the code on 200 simulated data sets which included noise. The smooth max actually stayed within 0.005 every time whereas the plain max overshot it by 0.85. A downside to using a smoothed max is that it draws a straight line through the data. If the real pattern is actually a curved line instead of a straight line, the answer would be off.
