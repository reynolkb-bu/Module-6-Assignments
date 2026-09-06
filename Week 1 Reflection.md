# Week 1 Reflection

## 1. In Coding Quiz 1, you are asked to find the distance of the farthest match in a set. Is this farthest match distance too far to be a meaningful match? How can you decide this?

Yes, it is too far.

It helps to think about what matching is really doing. Say you want to know how much a kitchen remodel adds to a home's price. You cannot remodel and not remodel the same house, so instead you find a similar house that was not remodeled and compare the two sale prices. The whole method rests on the two houses being similar. If the closest comparable home you can find is twice the size, the price gap tells you about size, not about the kitchen. Here, Z is the "size" we match on and Y is the "price" we compare.

So the question is whether the worst pair is still similar enough. Three ways to check:

**Compare the gap to the spread of the data.** The farthest match is 0.210217 apart. The standard deviation of Z is about 0.25, so that pair is roughly 0.84 standard deviations apart. A common rule of thumb is to allow no more than 0.2 standard deviations, which is about 0.05 here. This match is four times too far. The typical match is only 0.0133 apart, so this one is about 16 times worse than normal.

**Ask why it is so far.** That treated row has Z = 0.9884, and the closest control available is 0.7782, the largest control in the data. Nothing closer exists. This is not bad luck on one row either: 18 of the 48 treated rows have a Z higher than every single control, so all 18 get pushed onto that same edge control. It is the appraisal problem again, where the house you are pricing is bigger than anything that has sold nearby. You are guessing past the end of the data, and no smarter algorithm fixes it, because the comparable houses do not exist.

**See whether the far matches change the answer.** This is the most convincing test. Among the controls, Y rises about 1-for-1 with Z, so a 0.21 gap in Z drags about 0.21 of error into that comparison. Using all 48 matches gives an effect of 0.5434. Keeping only matches within 0.05 gives 0.5026, and within 0.02 gives 0.4997. All of them move toward 0.5, which is the true effect built into the data. The estimate moves once the bad matches are gone, which tells us they were the problem.

The tradeoff is that dropping those rows changes the question being answered. We are no longer measuring the effect for all treated rows, only for the ones that had a real comparison in the first place.

## 2. Invent your own type of matching similar to (A) and (B), which has a different way to pick the matches in X = 0.

My approach is kernel matching, also called distance-weighted matching. Like approach B, it uses every control within a set distance h. Unlike B, it does not treat them all as equal. Closer controls count for more.

For each treated row I measure how far away each control is in units of h:

    u = (Z_control - Z_treated) / h

and give that control a weight of 0.75 * (1 - u^2) when u is between -1 and 1, and 0 otherwise. Controls sitting right on top of the treated row get the full weight, and the weight falls off smoothly to 0 at distance h. I rescale the weights in each group so they add to 1, take the weighted average of those controls' Y values, and subtract that from the average Y of the treated rows.

The reason to do this is that A and B both ignore how close a match actually is. Back to the appraisal: approach A is an appraiser who uses exactly one comparable home and throws out the second-best one, even when the two are nearly identical. In this data it discards a control 0.014 away because another was 0.013 away. Approach B is an appraiser who averages every home within a mile and gives the one across the street the same say as the one at the far edge, treating a control 0.199 away just like one 0.001 away.

Kernel matching is what an actual appraiser does. It leans hardest on the closest comparables, lets distant ones fade out, and gives no weight at all past the cutoff, so no home jumps in or out of the calculation just because it moved an inch.

It also softens the problem from question 1. An edge control still gets used when it is the only option available, but it comes in with a small weight, so it does less damage to the estimate. Averaging several nearby controls instead of leaning on a single one also makes the comparison steadier.

The cost is having to choose h. A small h keeps the matches close but leaves few controls in each group, so the estimate gets noisy. A large h brings in more controls and smooths the estimate, but it lets worse matches back in and biases the result.
