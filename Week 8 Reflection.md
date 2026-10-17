# Week 8 Reflection

## 1. Questions 1 and 2: Inverse probability weighting (homework_8.1.csv)

**Question 1:** The ATE came out to 2.27, which is closest to B (2.3).

**Question 2:** The first three propensity scores were [0.84, 0.58, 0.71], which is A.

```python
def propensity_scores(df):
    model = LogisticRegression().fit(df[["Z"]], df["X"])
    return model.predict_proba(df[["Z"]])[:, 1]


def ipw_ate(df, p):
    treated = df["X"] == 1
    weights = np.where(treated, 1 / p, 1 / (1 - p))
    y_treated = np.average(df["Y"][treated], weights=weights[treated])
    y_untreated = np.average(df["Y"][~treated], weights=weights[~treated])
    return y_treated - y_untreated
```

A good real-world example is a job training program. People that are motivated are more likely to sign up, and they would also earn more money on their own. So if you're only comparing the people who sign up versus the ones that didn't, the job training program Gets credit for people’s own motivation.

IPW fixes this problem by giving more weight to people who are rare in their group. I use logistic regression to find each person's chances of signing up based on Z. `predict_proba` gives two columns, the chance of X = 0 and the chance of X = 1. I took the second one with [:, 1]. Each treated row gets a weight of 1/P and each untreated row gets 1/(1 - P). A person that is unmotivated and signs up counts for more because there aren't many unmotivated people in the sign-up group. Finally, I took the weighted average of Y in each group and subtracted them.

## 2. Questions 3 and 4: Mahalanobis matching (homework_8.2.csv)

**Question 3:** The matched ATE came out to 3.44, which is D (3.4).

**Question 4:** Its nearest untreated match is at (1.52, -1.28), which is D.

```python
def inverse_covariance(df):
    z = df[["Z1", "Z2"]].to_numpy().T          # 2 x N
    return np.linalg.inv(np.cov(z))            # invert the 2 x 2 covariance


for z, y in zip(treated[["Z1", "Z2"]].to_numpy(), treated["Y"]):
    d = np.array([mahalanobis(z, u, vi) for u in untreated_z])
    nearest = d.argmin()
    effects.append(y - untreated["Y"].iloc[nearest])
    distances.append(d[nearest])
    matches.append(untreated.iloc[nearest])
```

Every treated row searches every untreated row and takes the closest one. The same untreated row can be pick more then one time. This is matching with replacement.

## 3. Obstacles, challenges and insights

**Insight: I picked Mahalanobis instead of plain distance.** Let's imagine you're trying to find similar patients by weight and age. A 10-pound difference in weight isn’t that big of a deal, but a 10-year age gap is. If you use plain distance, it treats them as the same. Manhalanobis scales each variable by how spread out it is.

**Insight: Common support.** The treated item at (2.70, 0.54) has no untreated item close to it. It’s like trying to compare a 7-foot basketball player to someone that is short and didn’t play. Since the closest match is so far away, that comparison is not as trustworthy as the others.

**Obstacle: I got Q4 wrong the first time.** I actually got question 4 wrong initially. I read “the nearest Z1 and Z2 values of the treated item" as the treated item's own values, (2.70, 0.54), and picked A because it was the closest option. The question was really asking for is the values of its nearest match, which is D. 

**Challenge: Getting the covariance matrix the right shape.** The quiz said to put all the Z1 and Z2 values into a 2xN matrix. My data has one row per item and one column for Z1 and Z2, which is N x 2. If you don’t flip it with `.T`, `np.cov` compares all 1,000 rows to each other instead of comparing the two columns and gives a 1000 x 1000 matrix. Adding .T gives the 2 x 2 matrix that the quiz asked for.
