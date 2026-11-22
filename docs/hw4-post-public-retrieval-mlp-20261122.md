# Post-Public retrieval MLP: two bounded negative checks

Both feature views failed the predeclared admission gate. No full refit or submission CSV was produced. Context04 OLD mean AP is 0.2675064148; raw GPU TabICL94 is 0.2810693628.

| Feature view | Matched MLP | Retrieval MLP | MLP50 / ICL50 | Retrieval50 / ICL50 |
| --- | ---: | ---: | ---: | ---: |
| Own22 | .2636843168 | .2651728688 | .2776084018 | .2769782996 |
| Compact94 | .2554255827 | .2459042481 | .2730674434 | .2682784194 |

Retrieval improved on its own22 control by .0014885520 but remained below context04. Compact94 was one explicitly authorized matched feature ablation after the own22 outcome; it is exploratory follow-up, not independent confirmation. No further widths, neighbors, weights or optimizer settings were tried.

## Fixed method and causal boundary

This is a simplified retrieval MLP inspired by [TabR's paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html) and [official repository](https://github.com/yandex-research/tabular-dl-tabr/blob/main/README.md), not an official TabR reproduction. It uses no Faiss, pretrained weights or new dependencies.

Numeric inputs receive FIT-only median imputation, normal quantiles with at most 1000 quantiles, clipping to [-5,5], and missing flags. Four FIT-only categorical vocabularies use 4-dimensional embeddings; unknown values map to 0. Own22 has 18 numeric fields and input width52; compact94 has90 numeric fields and input width196. Both use two64-wide encoder layers, GELU/dropout.1, LayerNorm and a scalar head. Control and retrieval instantiate identical parameter layouts and initial weights; the control disables retrieval.

Retrieval uses exact torch k16 over detached FIT embedding caches refreshed each epoch. Selected neighbors are re-encoded with gradients. Attention combines normalized-key distance, feature differences and embeddings of FIT labels. Every training or prediction query excludes the same physical site and candidate quarters greater than or equal to its own quarter. Masked top-k slots have zero attention; empty history yields exactly zero retrieval aggregate. IDs and dates are mask metadata, never encoder inputs.

AdamW lr.001/weight decay.0003, batch256, clip1, seed42, CPU2 and float32 were fixed. Each algorithm has three previous-quarter inner fits (max30 epochs/patience6, earliest maximum AP) followed by three fresh outer fits at those fixed epoch counts. Raw admission requires improvement over context04; a fixed50/50 probability mixture must improve over ICL94. Both require at least2 nonnegative quarter deltas, worst delta≥−.005 and nonnegative2023Q4 delta. All candidates failed.

## Actual compute and validation

There were **28 real-data optimizer jobs**:24 formal fits (12 inner +12 outer) and4 separate timing-only probe fits. These probes trained for two epochs each, called AP zero times, and were discarded before the formal fits. They are not synthetic QA. There were0 full fits and0 exports.

| View | Two-epoch control / retrieval jobs | Conservative projected maximum | Probe peak RSS | Complete run |
| --- | --- | ---: | ---: | ---: |
| Own22 | .628s /1.168s |275.43s |729,219,072 bytes |54.72s |
| Compact94 | .786s /1.453s |304.29s |758,546,432 bytes |60.60s |

Nine unique targeted checks were repeated for both views: matched initialization; zero empty-context aggregate; fewer-than-k masks; future/same-site exclusion; forbidden-label mutation invariance; finite gradients; key/label gradients; exact state reload; and train-only unknown-category preprocessing. All28 saved model forwards repeated exactly after state reload. The combined audit also verified every serialized preprocessor's FIT-only medians, quantile endpoints and vocabularies, all chronological row hashes, cached gate arithmetic and fixed mixture formulas without further AP or model predictions.

The original88 frozen/source/scorer files, single recent diagnostic and four published native32 files stayed unchanged. New2024Q2/Q3 AP and Kaggle uploads are0. OLD development is heavily reused and both checks followed Public feedback; no fresh-holdout claim applies.

## Reproduction evidence

Source SHA256:

- `src/hw4_inspections/retrieval_mlp.py`: `eb0edc3e4dcf610e1fa9c550a3745979d1d312637281040788c1e4a3795fa16c`
- `scripts/run_hw4_retrieval_mlp.py`: `131bf00a6c489f17c9d1a57b9d0f4561736ffdcf27abe19a260ea3e90ddb15b7`
- `scripts/run_hw4_retrieval_mlp94.py`: `cbc57b209771d9d33eee941c0ebd3629fb1f4fc6330b270f5bf098aca3e38cb2`

Runners use the recorded local workspace paths and verified cache/OOF inputs in each protocol. Every run preserves its freeze, executed source copies, timing probe, inner choices,14 checkpoints, OLD predictions and metrics under `analysis/hw4-post-public-20261122/{retrieval-mlp,retrieval-mlp94}/`.

Combined receipt: `analysis/hw4-post-public-20261122/retrieval-mlp-combined-verification.json`, SHA256 `e9a52adb4c1241560f1b7a306fa4f17f55c4ea7e43151e0207281ea01e79ade0`.
