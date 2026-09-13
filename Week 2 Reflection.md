# Week 2 Reflection

## 1. Invent an example situation that would use fixed effects.

An example situation that would use fixed effects would be a coffee chain. Let's say the chain has 30 different locations. They roll out a new menu and want to know whether the new menu accounted for an increase in sales week over week.

The issue with this is that the stores were never comparable. For example, a location at an airport would sell a lot more than a location at a mall in a town in the middle of nowhere.

Really, what you would want to do is compare the stores across locations, not the actual menu. Using fixed effects would fix this issue by giving each store its own baseline. Each store would get its own intercept and the shared menu would be estimated on top of that. The question then becomes: did each store beat its own weekly sales? 

## 2. Bootstrap the variance in the mean of a Pareto distribution. Explain what you had to do. As the sample size grows, what happens to that variance?

As the sample size grows, variance gets smaller. For example, think of Google reviews. A store with only a few Google reviews would be greatly impacted by one bad review or one good review. However, the more reviews a store gets, the less each individual review moves their average. This relates to Pareto values because the average becomes more stable. 

To bootstrap the variance of the mean of a Pareto distribution, I took one sample from the distribution. After that, I made 5,000 new samples by randomly picking values, allowed repeating values, and then took the average of each. The variance of the mean is how much those 5,000 averages vary.

Every time the sample size increased by 4x, the variance dropped by about 25%, which makes sense. This means the variance shrinks at about 1/n, which matches the Pareto theory. The one issue I found is that Pareto data sometimes will have big values, meaning a single run could seem scattered. Therefore I had to repeat each run 40 times and average those runs out.
