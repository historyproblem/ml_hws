# Native CatBoost MAP: stopped at resource gate

After Public11/12 the user authorized further research. A bounded native
CatBoost `YetiRank:mode=MAP` experiment was fixed before real fitting:
full346 and compact94, one seed42, cap800 with100/300/500/800 checkpoint
selection on the strict preceding inner quarter; at most6inner+3outer+1full.
Only reused OLD3 AP would have been scored. The full gate additionally required
nonnegative2023Q4 delta and worst-quarter delta no worse than−.005.

The installed CatBoost1.2.10 accepted MAP with MVS.8 on a synthetic64-row,
two-query-group dataset. Five native trees and saved-model reload were exact.
No Bernoulli substitution or environment change was necessary. The official
[ranking documentation](https://catboost.ai/docs/en/concepts/loss-functions-ranking)
supports non-Classic YetiRank MAP on CPU.

The first real inner fit used9293training rows,1808queries,full346 and800trees.
It did not complete within the predeclared30-second hardlimit. An external
watchdog terminated the process with SIGTERM; total process elapsed31.67s.
The branch stopped immediately, with no further model fits or parameter retry.

**Actual scope: one interrupted real fit attempt, zero completed real models,
zero AP values, zero outer/full fits and zero exported CSV.** This is a resource
outcome and says nothing about MAP forecast quality. Native32 sources, prior
frozen predictions and recent metrics were not changed.

The saved full346 view preserved the reference10 categorical fields and numeric
quarter. The planned compact94 view explicitly categorized quarter; it never
ran. The executed script/freeze are preserved, with actual scope in
`analysis/hw4-post-public-20261122/catboost-map/complete.json`, and the external
watchdog receipt beside that directory.
