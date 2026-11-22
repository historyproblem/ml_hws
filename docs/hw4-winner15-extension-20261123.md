# Three fixed extensions of winner15

All three OLD-only extensions failed the predeclared gate. No models were fitted, no bootstrap or full forecast was run, and no new submission CSV was produced. Winner15 remains unchanged: equal50 log-odds of the arithmetic three-seed context CatBoost probabilities and native4 TabICL94 probabilities. Its user-observed Public score is0.35716; this motivated exploratory follow-up, not a new independent holdout.

Let `L(p)=logit(clip(p,1e-6,1−1e-6))`. The only new formulas were frozen before their metrics:

| Formula | Weights inside expit | OLD mean AP | Δ vs15 | Nonnegative quarters | 2023Q4 Δ |
| --- | --- | ---: | ---: | ---: | ---: |
| Winner15 reference | .5L(C3)+.5L(I4) | .2766506669 | — | — | — |
| A | .5L(C3)+.5L(I32) | .2757577183 | −.0008929486 |0/3 |−.0014497010 |
| B | .5L(C3)+.25L(I4)+.25L(I32) | .2763294723 |−.0003211946 |1/3 |+.0000120292 |
| C | .5L(C10)+.5L(I4) | .2759279528 |−.0007227141 |1/3 |+.0005024884 |

`C3` is the mean of context CatBoost seeds42/17/2026; `C10` averages those three and the previously frozen seeds7/23/101/123/456/777/1000. Native4/32 use their verified compact94 forecasts. All formulas retain50% CatBoost; no coefficient, epoch, seed subset or additional blend was searched.

Admission required mean improvement over15, at least2 nonnegative OLD-quarter deltas, worst delta≥−.002 and nonnegative2023Q4 delta. All fail the mean criterion; B/C also have only one nonnegative quarter. These small differences are not significance claims. Per the frozen protocol, bootstrap was skipped and the seven possible new full CatBoost fits for C were not launched.

## Input and result verification

The5815 OLD rows and all metadata were aligned exactly. Reference15 matches its original signed `duo50` probabilities and cached per-quarter AP exactly; its2246-row full forecast also reproduces exactly. All21 old bag10 checkpoint markers, model/prediction hashes and fixed500-tree identities were verified. Stored seven-seed arrays reconstruct C10 exactly; no old fit or predictor replay was performed. All four native32 context identities and NPY-to-CSV prediction hashes match their accepted source protocol. Full parent sample order and probability ranges were checked without producing new candidates.

The new screen contains exactly three candidate formulas plus the reference. Cached summary/gate arithmetic was verified without additional AP calls. A guarded completed resume performed0 scoring, inference or fitting and preserved every byte. The previous88 frozen/source/scorer files, the one recent diagnostic and winner15 CSV remained unchanged. New2024Q2/Q3 AP and Kaggle uploads are0.

## Reproduction

New source: `scripts/run_hw4_winner15_extension.py`. Outputs are under `analysis/hw4-winner15-extension-20261123/zero-fit/`; parent-level Public observation and execution metadata are separate. The script freezes before metrics and refuses changed completed evidence.

```bash
python scripts/run_hw4_winner15_extension.py \
  --workspace '/Users/khlystov/Documents/ChatGPT/ML ДЗ' \
  --out '/Users/khlystov/Documents/ChatGPT/ML ДЗ/analysis/hw4-winner15-extension-20261123/zero-fit' \
  --stage screen
```

Freeze signature: `4c3fc0eaac1b237965185a88d916cfbe2eec319234e9a35504e3b7dd7783c6d9`. Freeze SHA256: `b0e534c34b0c9003b958385be26c997b2fbdf3dc3c45ac4712e3dd03dfb6e4aa`.

`screen-complete.json`, `input-verification.json`, `verification.json`, the executed source copy and the OLD tables preserve the exact evidence. The failed gate is final for this three-form experiment; no follow-up weight refinement was performed.
