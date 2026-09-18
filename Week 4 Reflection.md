# Week 4 Reflection

## 1. The Coding Quiz gives two options for instrumental variables. For the second item (dividing the range of W into multiple ranges), explain how you did it, show your code, and discuss any issues you encountered.

I split the data into 20 groups based on W, so everyone in a group has about the same W. In each group, I found how much Y changed when Z went from 0 to 1, and divided it by how much X changed. Then I took the average across all 20 groups. I got 1.51, which is almost the same as the 1.56 from the first option.

```python
def effect(group):
    m = group.groupby("Z")[["X", "Y"]].mean()
    return (m.loc[1, "Y"] - m.loc[0, "Y"]) / (m.loc[1, "X"] - m.loc[0, "X"])

groups = pd.qcut(df["W"], 20)
print(np.mean([effect(g) for _, g in df.groupby(groups, observed=True)]))  # 1.51
```

The issue I ran into was that when I first split W into equal-sized chunks, the groups at the edges only had a few people in them. Those tiny groups gave crazy answers, like -45. I fixed it by using `pd.qcut`, which puts the same number of people in every group.

## 2. Plot the college outcome (Y) vs. the test score (X) in a small range of test scores around 80. On the plot, compare it with the Y probability predicted by logistic regression.

![College admission vs. test score near 80](week4_reflection.png)

Y is either 0 or 1, so plotting it directly would just show two lines of dots. Instead, I grouped students with similar scores together and plotted the percent of each group that got into college. Those are the gray dots.

The blue dashed line is a normal logistic regression. It can only draw a smooth curve, so it misses the jump at 80. The orange line is a logistic regression that is allowed to jump at 80, and it matches the dots much better. In dataset a, for example, the chance of getting in jumps from about 31% to 60% right at a score of 80.
