# TabICL diversity after winner15

Winner15 uses fixed equal50 logit mixing of context CatBoost04 and native4
TabICL94; its observed Public score0.35716 exceeds baseline0.35212. This new
bounded experiment changes the TabICL Monte Carlo ensemble, preserving winner15
and all earlier predictors. This is post-Public exploratory development.

Exactly two new native32 seeds17/2026 each predict OLD3 and full test:8contexts.
The accepted seed42 native32 outputs are reused byte-exact from their existing
directory, with no repeated42 inference. Features compact94, full past contexts,
public checkpoint and native package versions remain the same. Every job records
its seed, actual32 transformed views, contexts and prediction hashes. The first
complete OLD forward repeats exactly for each seed. Only OLD metrics are scored;
2024Q2/Q3 are neither predicted nor scored. There is no neural weight update.

Two recipes are fixed before any new-seed quality is observed. In both, clip is
1e-6 and the final C/I logit coefficients remain exactly50/50:

1. I=probability mean of native32 seeds42,17,2026.
2. I=.5*native4seed42+.5*the same native32 three-seed mean.

No best-seed choice, alternative weights or further pooling variants are allowed
in this run. Each final blend is compared with winner15 OLD predictions. Admission
requires positive equal-quarter mean delta, at least2nonnegative quarters, worst
delta at least−.002, and nonnegative2023Q4 delta. These reused OLD quarters are not
independent confirmation; Public improvement is unknown. Candidate CSV is exported
only for a predeclared final blend that passes all four admission conditions.

The wrapper reuses the existing isolated environment and public checkpoint.
It does not install packages, change the host environment, download assets or
upload to Kaggle. The accepted original seed42 six files and all local winner15
parents are pinned by SHA. The result archive contains six regular known files;
the consumer bounds gzip/tar expansion and validates sources, data, settings,
contexts, seeds, forecasts, metrics and both fixed formulas before writing results.

Measured seed42 native32 OLD3/full runtime was281.5s including one complete repeat.
Eight new contexts are estimated at9–11minutes on the same V100, not yet measured.
All earlier14GPU producer files, four native32 files,87frozen parents and winner15
remain unchanged. Native CPU real-checkpoint QA uses only synthetic data.
