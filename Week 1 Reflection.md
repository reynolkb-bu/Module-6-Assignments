# Week 1 Reflection

## 1. In Coding Quiz 1, you are asked to find the distance of the farthest match in a set. Is this farthest match distance too far to be a meaningful match? How can you decide this?

Yes, it is too far. The farthest match is 0.210217 apart, but the standard deviation of Z is only about 0.25, so that pair is roughly 0.84 standard deviations apart on the one variable we are matching on. A common rule of thumb is to allow no more than 0.2 standard deviations, or about 0.05 here, so this match is four times too far. The typical match is only 0.0133 apart, making this one about 16 times worse than normal.

It also helps to ask why it is so far. That treated row has Z = 0.9884 and gets paired with the largest control Z in the data, 0.7782. It is not an unlucky pairing: 18 of the 48 treated rows have a Z higher than every control, so they all get pushed onto that same edge control. That is guessing past the end of the data, and no better algorithm can fix it because the comparable rows do not exist.

The clearest test is to see how much the far matches change the answer. Y rises about 1-for-1 with Z among the controls, so a 0.21 gap adds about 0.21 of error. Using all 48 matches gives an effect of 0.5434; keeping only matches within 0.05 gives 0.5026, and within 0.02 gives 0.4997. All move toward 0.5, the true effect in the data. Since the estimate shifts that much, the far matches were the problem. The catch is that dropping them means we only measure the effect for treated rows that have real comparisons.

## 2. Invent your own type of matching similar to (A) and (B), which has a different way to pick the matches in X = 0.

My approach is kernel matching, or distance-weighted matching. It keeps every control within a bandwidth h, like approach B, but weights each one by how close it is. For each treated row I compute u = (Z_control - Z_treated) / h and give each control the weight 0.75 * (1 - u^2) when u is between -1 and 1, and 0 otherwise. I rescale the weights in each group to add to 1, take the weighted average of those controls' Y values, and subtract the average of those from the average Y of the treated rows.

The point is that A and B both ignore how close a match actually is. A throws away a control 0.014 away even though the winner was 0.013. B counts a control 0.199 away just as heavily as one 0.001 away. Kernel matching sits in between: close controls carry most of the weight, far ones fade out, and the weight reaches 0 at the edge, so nothing jumps in or out suddenly.

This also softens the problem from question 1. An edge control still gets used when it is the only option, but with a small weight, so it does less damage. Averaging several nearby controls also makes the comparison value less jumpy than relying on one. The downside is choosing h: small h is more accurate but noisier, large h is smoother but more biased.
