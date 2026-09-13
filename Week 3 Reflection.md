# Week 3 Reflection

## 1. In the event study in Coding Quiz 3, how would we go about testing for a change in the second derivative as well?

Add a squared time term, and let it switch on at the event, the same way we did for the slope. With s = time − 50 and post = 1 after the event, the model is y = a + b·s + c·s² + d·post + e·s·post + f·s²·post. Here d is the jump in value, e is the jump in slope, and f measures the jump in curvature.

The test is whether f is different from zero, using its t-statistic. I ran this on `homework_3.1.csv` (program: `week3_reflection.py`). None of the three series showed a change in curvature (all p-values above 0.15), so the quiz answers still hold.

Two cautions. Every extra term uses up data, and there are only 50 points on each side of the event, so the estimates get noisier. A squared term can also bend at the ends of the data in ways that look like a jump, so it's safer to fit only the points near the event.

## 2. Create your own scenario that illustrates differences-in-differences. Describe the story behind the data and show whether there is a nonzero treatment effect.

A software company launches a new onboarding checklist to reduce support tickets. It goes live for US customers in month 7, but the EU launch is delayed by a privacy review. That delay gives us a control group: EU customers keep the old onboarding. Comparing the US before and after isn't enough, because tickets were already falling everywhere as the product improved.

I simulated 200 customers per region over 12 months, with a true checklist effect of −0.8 tickets per month (program: `week3_reflection.py`). US tickets fell by 1.34 after launch, but EU tickets also fell by 0.55 without the checklist. The difference-in-differences is −1.34 − (−0.55) = −0.78. Looking only at the US would have overstated the effect by about 70%.

The regression gives the same −0.78, with a 95% confidence interval of −0.89 to −0.67. That interval excludes zero, so the treatment effect is clearly nonzero, and it contains the true −0.8. Before launch, both regions' ticket trends moved together (p = 0.86), which supports the parallel-trends assumption the method depends on.
