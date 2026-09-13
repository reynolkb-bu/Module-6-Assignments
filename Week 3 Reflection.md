# Week 3 Reflection

## 1. In the event study in Coding Quiz 3, how would we go about testing for a change in the second derivative as well?

The best way I thought of to go about testing for a change in the second derivative is to think of a car. The value is where the car is at a certain point in time. The first derivative would be its speed, and the second derivative would be whether it's speeding up or slowing down.

The quiz tested whether the car suddenly moved to a new spot in time and whether its speed changed. Testing the second derivative is simply asking whether the car sped up or slowed down at a certain point in time. In order to test it we can add an additional term to the regression equation. The term would be a squared term that only applies after the event. If that term is different from zero, that means the curve changed at that event. 

I then ran this on the quiz data. In my `week3_reflection.py` file, none of the three datasets actually showed a change in the second derivative, meaning the answer still holds. 

## 2. Create your own scenario that illustrates differences-in-differences. Describe the story behind the data and show whether there is a nonzero treatment effect.

My own scenario that illustrates differences in differences would be a software company that adds an onboarding checklist. They add this checklist because they're hoping to reduce the number of support tickets. U.S. customers get this onboarding checklist right away, but EU customers don't get it until later. 

Support tickets went down in the U.S. after the onboarding checklist launched, but they also went down in the EU, which never got the checklist. This means the drop in support tickets was not caused solely by the checklist.

In my simulated data, the number of U.S. support tickets dropped by 1.34 per month, and EU tickets dropped by 0.55 per month. The checklist did have a real effect: 1.34 - 0.55 = about 0.78 fewer support tickets per month. This means that there is a non-zero treatment effect since the onboarding checklist actually reduced the number of support tickets. 
