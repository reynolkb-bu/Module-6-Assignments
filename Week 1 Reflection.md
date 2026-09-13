# Week 1 Reflection

## 1. In Coding Quiz 1, you are asked to find the distance of the farthest match in a set. Is this farthest match distance too far to be a meaningful match? How can you decide this?

Yes, a distance of the farthest match in a set is too great to be a meaningful match. The reason is that matching usually only works if the pair is similar. For example, pricing a kitchen remodel by comparing a remodeled house to one that has not been remodeled. This would be like comparing a house that is twice as large. The pricing difference would be about the size of the house, not the actual kitchen.

One way to decide is to compare the gap in the data spread. The furthest match is about 0.21 apart, which is about 0.48 standard deviations. A common rule of thumb we can use allows for about 0.2 standard deviations. The typical match here is only 0.013 apart. This is due to 18 of the 48 treated rows having a larger Z than every control, so they all get matched to the same edge control.

The way we can be certain about this is to check whether the far matches actually change the answer. If we use all the matches, the effect is 0.543. If we keep only the matches within 0.05, it gives us 0.503. If we keep all the matches within 0.02, it gives us 0.500. The estimate improves once the bad matches are dropped, which proves that they were indeed the problem.

## 2. Invent your own type of matching similar to (A) and (B), which has a different way to pick the matches in X = 0.

The approach I would use that is similar to A and B would be weighted matching. Similar to approach B, I would take every control within a set of each treated row. The difference is that closer controls count more than the farther ones. A control right next to the treated row would be weighted more, whereas a control near the edge would not be weighted as much.

When we use approach A, it picks the control that is the closest, regardless of other controls being almost as close as the one that gets picked. When using approach B, it will look at every nearby control. Let's say there is one at 0.0 and another at 0.9. It would weight these equally, which doesn't really make sense.

Weighted matching is a balanced way of doing it. It would be like pricing a house where you look at houses that are nearby but put more weight on the ones that are very similar to each other.
