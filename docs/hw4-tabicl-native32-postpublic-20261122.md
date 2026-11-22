# Native32 TabICLv2 extension after Public feedback

The user explicitly reopened model computation after observing Public scores
0.34642 for compact94 native4 and 0.34242 for the GAM75/TabICL94 logit blend.
The incumbent context CatBoost remains0.34646. This is exploratory research
after Public feedback; old quarter results are reused development evidence.

One configuration is fixed before execution: exactly the same compact94 view,
public TabICLv2 checkpoint, full historical contexts, seed42, and FP32 native
inference, increasing the native ensemble from4 to32 transformed views. There
is no learned neural weight update and no parameter sweep. Neither original
producer files nor previous frozen predictions/recipes are changed. Existing
native4 predictions are not treated as a reusable prefix of native32.

Only four jobs run:2023Q3,2023Q4,2024Q1 and the final2024Q4 test. Q2/Q3 are not
predicted or scored. Three OLD AP values are recorded. The full test uses all
21017 official training rows. Data, source and checkpoint SHA plus context IDs,
actual ensemble setting, query memory fallback and repeated exact first-fold
forward are recorded. CUDA is required for real execution; there is no silent
CPU fallback. Package versions and public checkpoint stay pinned.

The completed original V100 compact94 jobs with4 components took11.90,8.39,
9.80 and14.47seconds for OLD3/full; a linear estimate for32 is several minutes,
with setup and preprocessing additional. This estimate is not a measured32
runtime. Mac own22 four-component full inference previously took337seconds,
so V100 is preferred for the larger94-column32-component calculation.

`scripts/run_hw4_tabicl_ensemble32.sh` reuses the already completed V100
TabICL environment and checkpoint under `analysis/.hw4-tabicl-v2-assets`.
It does not install dependencies, update the host environment or download
weights. The wrapper holds the same TabICL environment lock. The return archive
contains exactly six regular basenames, with no model, executable or recent
predictions. `scripts/analyze_hw4_tabicl_ensemble32.py` reads bounded bytes,
validates hashes, IDs, context cutoffs and OLD AP, and exports one full test CSV.
Neither producer nor consumer performs a Kaggle upload.

QA before publication: Bash syntax and Python compilation passed. A native CPU
probe with the actual public checkpoint, two components,64 synthetic training
rows,16 queries and8 mixed-type columns passed repeated exact predictions.
The synthetic score vector SHA is
`a8ff7873069369e9213cbeeec7fd21f798e89ceacfaf5296e98604a3932dddc2`.
No real HW4 native32 inference had run at this pre-execution QA point.
