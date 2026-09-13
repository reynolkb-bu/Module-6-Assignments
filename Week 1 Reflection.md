# Week 1 Reflection

## 1. In Coding Quiz 1, you are asked to find the distance of the farthest match in a set. Is this farthest match distance too far to be a meaningful match? How can you decide this?

Yes, a distance of the farthest match in a set is too great to be a meaningful match. The reason is that matching usually only works if the pair is similar. For example, pricing a kitchen remodel by comparing a remodeled house to one that has not been remodeled. This would be like comparing a house that is twice as large. The pricing difference would be about the size of the house, not the actual kitchen.

One way to decide is to compare the gap in the data spread. The furthest match is about 0.21 apart, which is about 0.48 standard deviations. A common rule of thumb we can use allows for about 0.2 standard deviations. The typical match here is only 0.013 apart. This is due to 18 of the 48 treated rows having a larger Z than every control, so they all get matched to the same edge control.

The way we can be certain about this is to check whether the far matches actually change the answer. If we use all the matches, the effect is 0.543. If we keep only the matches within 0.05, it gives us 0.503. If we keep all the matches within 0.02, it gives us 0.500. The estimate improves once the bad matches are dropped, which proves that they were indeed the problem.

## 2. Invent your own type of matching similar to (A) and (B), which has a different way to pick the matches in X = 0.

My approach is kernel matching. Like approach B, it uses every control within a distance h of each treated row. Unlike B, closer controls count for more. Each control gets a weight that is highest when it sits right on the treated row and falls smoothly to zero at distance h. I compare the treated row's Y to that weighted average of control Y values.

This fixes a weakness in both A and B: neither pays attention to how close a match actually is. Approach A uses one control and throws away a second one that is almost as close. Approach B treats a control 0.199 away the same as one 0.001 away. Kernel matching leans on the closest controls, the way a good appraiser leans on the most similar houses.

The cost is choosing h. A small h keeps matches close but leaves few controls, so the estimate is noisy. A large h brings in more controls and smooths the estimate, but lets worse matches back in.
