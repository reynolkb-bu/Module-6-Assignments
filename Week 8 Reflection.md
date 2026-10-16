# Week 8 Reflection

## 1. Average treatment effect with inverse probability weighting (homework_8.1.csv)

The ATE came out to 2.27, which is closest to B (2.3). The first three propensity scores were [0.84, 0.58, 0.71], which is A.

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

First, I fit a logistic regression where Z predicts X. The propensity score is the chance that a row gets treated given its Z. `predict_proba` gives back two columns, the chance of X = 0 and the chance of X = 1, so I took the second column with `[:, 1]`. If I took the first column by mistake, every score would be flipped and the answer would be wrong.

Next, I made the weights. A treated row gets 1/P and an untreated row gets 1/(1 - P). The idea is that a treated row with a low propensity score is rare, since it got treated even though rows like it usually don't. That means it stands in for a lot of rows that didn't get treated, so it gets a bigger weight. The same goes for an untreated row with a high propensity score.

Last, I took the weighted average of Y for each group and subtracted them. I used `np.average` with weights, so each group is divided by the sum of its own weights instead of by the total number of rows. I chose this because the weights don't add up to exactly N in a real sample. Dividing by N instead gave 2.24, so both ways still land on B.

## 2. Mahalanobis matching (homework_8.2.csv)

The matched ATE came out to 3.44, which is D (3.4). The treated row with the least common support is at Z1 = 2.70, Z2 = 0.54. Its nearest untreated match is at Z1 = 1.52, Z2 = -1.28, which is D (1.5, -1.3).

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

Every treated row searches all of the untreated rows, so the same untreated row can be picked more than once. That is how matching with replacement works. For Q4, I kept the distance of each treated row to its best match and the match itself. Then I took the treated row with the largest distance. That row has the least common support because even its closest untreated neighbor is far away. The question asks for the Z1 and Z2 of that nearest neighbor.

## 3. Obstacles, challenges and insights

**Weighting moved the answer a lot.** If you just compare the averages of the treated and untreated rows, the effect is 3.04. IPW brings it down to 2.27. Rows with high Z were more likely to be treated and also had higher Y, so the plain comparison gave the treatment credit for Z. I also checked this with a regression of Y ~ X + Z and got 2.27, so both methods agree.

**sklearn regularizes by default.** `LogisticRegression` adds a penalty unless you turn it off, which shrinks the coefficients a little. I was worried this would change the propensity scores, so I ran it again with no penalty. The ATE went from 2.274 to 2.268, so it didn't change the answer. With only one Z and 1,000 rows the penalty barely does anything, but on a smaller dataset it could matter.

**Large weights.** The propensity scores ran from 0.05 to 0.93, so the biggest weight was about 12.7. That means one row counts as almost 13 rows. If a score got very close to 0 or 1, one row could swing the whole answer. This data was fine, but it showed me why common support matters for IPW too, not just for matching.

**Why Mahalanobis.** Z2 is more spread out than Z1 (sd 1.36 vs 0.99). With plain distance, a gap in Z2 would count the same as a gap in Z1 even though a gap in Z2 is more normal. Mahalanobis divides by the spread and also accounts for Z1 and Z2 moving together. In this data, the plain distance gave 3.48 instead of 3.44, so it was close, but it would matter more if the variables were on very different scales.

**Reusing matches.** The 483 treated rows were matched to only 172 different untreated rows. A small number of untreated rows did a lot of the work, which makes the answer depend heavily on them.

**What the quiz calls the ATE.** Only treated rows were matched in Q3, so the 3.44 is really the effect on the treated (ATT), not the effect across everyone. The two can be different when the treated rows don't look like the full population. The IPW number in Q1 weights both groups, so it is the true ATE.

**Q4 wording.** I got Q4 wrong the first time. I read "the nearest Z1 and Z2 values of the treated item" as the treated item's own values, (2.70, 0.54), and picked A because it was the closest option. The question was actually asking for the values of its nearest match, (1.52, -1.28), which is D. The warning sign was that none of the options really matched my number. A was 0.77 away while my other answers matched almost exactly. I measured the distance to each option and took the smallest instead of rereading the question. The lesson I took away is that when an answer only loosely fits the choices, I should check that I'm reporting the right thing before picking the closest option.
