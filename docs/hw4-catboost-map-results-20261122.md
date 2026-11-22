# Native CatBoost MAP: completed exploratory experiment

The initial30-second resource trial was cancelled before any real model or AP
completed. Because over an hour remained, the allocation was explicitly amended
before observing quality:90seconds per fit and15minutes total. Loss, both views,
seed42,800-tree cap and100/300/500/800 checkpoint grid were unchanged. The
cancelled attempt, original executed source and its freeze remain preserved.

The new `catboost-map-budget90` run completed6inner and3outer fits in544.67s.
Inner durations were46.2–68.1s, outer durations40.2/69.6/86.0s; the external
guard did not stop the retry. Every outer saved-model forward reproduced
exactly. Native saved configurations confirm `YetiRank:mode=MAP`,MVS,depth5,
seed42 and the selected500/800/800trees. Full346 preserves the incumbent's10
native categories and numeric quarter; compact94 has four categories including
quarter. The chosen view/count uses only the strict previous inner quarter.

| Model | 2023Q3 AP | 2023Q4 AP | 2024Q1 AP | Mean AP | Delta to context |
|---|---:|---:|---:|---:|---:|
| Context CatBoost3seed | .276520 | .273840 | .252158 | .267506 | 0 |
| Native MAP | .270420 | .271343 | .243473 | .261745 | −.005761 |
| Context/MAP rank50 | .277152 | .274199 | .247376 | .266242 | −.001264 |

The raw MAP model is below context on all three OLD quarters. The fixed rank50
mixture improves two quarters and satisfies the special2023Q4 and worst-delta
conditions, but fails the positive mean gate. Therefore **zero full refits and
zero submission CSV** were produced. No recent2024Q2/Q3 metrics, new blend
weights or Kaggle uploads were computed.

These reused OLD quarters support an exploratory negative decision for this
specific fixed configuration. They do not establish that MAP optimization is
generally inferior. Across both allocation attempts:10real fit attempts,
9completed and1resource-interrupted; the separate synthetic toy is excluded.

The zero-fit/no-AP follow-up audit verified5815official old IDs, all6inner
checkpoint choices, view selection, exact rank vectors, summary gates and
saved native configurations. Its signed JSON is
`analysis/hw4-post-public-20261122/audit/catboost-map-completion-review.json`,
SHA256 `bd249d144104bffefd3ad7b3dfc79c21f7a61ad5a51bfc9465f5cc475618b79e`.

[Official CatBoost ranking loss documentation](https://catboost.ai/docs/en/concepts/loss-functions-ranking).
