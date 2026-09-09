# #3521 — does the corrected driver produce a usable profile?

Device `cuda`, profile cell key `cuda+cuml` — the arms below are only meaningful if the profiles' cells resolve under that key (`cell_keys` derives the cuML suffix from the *live* backend, so a `cuda+cuml` profile read on a host without it silently becomes the shipped arm). Each profile is scored on the *other* leg's runs;
`shipped` is scored on both. `bar` is the fraction of the progress bar budgeted to
the wrong step (0 is perfect, 1 is every second in the wrong slot); `step` is the
median per-step relative error over steps that took at least
0.05 s.

| arm | task | branch measured | runs | bar error | step error |
|---|---|---|---:|---:|---:|
| new | dataset_load | embed=cached | 4 | 0.66 | 0.13 |
| new | dataset_load | embed=fresh | 12 | 0.01 | 0.06 |
| new | dataset_open | coverage=restored | 32 | 0.62 | 0.13 |
| new | dataset_stage | embed=cached | 8 | 0.71 | 0.91 |
| new | dataset_stage | embed=fresh | 8 | 0.00 | 0.04 |
| new | text_sort | plain | 96 | 0.84 | 0.25 |
| old | dataset_load | embed=fresh | 16 | 0.09 | 0.06 |
| old | dataset_open | coverage=rebuilt | 16 | 0.94 | 0.99 |
| old | dataset_open | coverage=restored | 32 | 0.00 | 0.20 |
| old | dataset_stage | embed=fresh | 16 | 0.35 | 0.15 |
| old | text_sort | plain | 96 | 0.85 | 0.22 |
| shipped | dataset_open | coverage=rebuilt | 16 | 0.15 | 0.73 |
| shipped | dataset_open | coverage=restored | 64 | 0.84 | 0.81 |
| shipped | dataset_stage | embed=cached | 8 | 0.85 | 0.92 |
| shipped | dataset_stage | embed=fresh | 24 | 0.67 | 0.99 |
| shipped | text_sort | plain | 192 | 0.80 | 0.52 |

## Within-leg holdout

Half of each leg's own reps fit the profile, the other half score it. The
cross-leg table above never asks an arm to predict a branch its own leg
measured and the other did not - in particular it never asks `new` to pace an
atlas rebuild, which is the branch it exists for. Fits here stand on half the
rows, so read the ranking, not the third digit.

| arm | task | branch measured | runs | bar error | step error |
|---|---|---|---:|---:|---:|
| new | dataset_load | embed=fresh | 8 | 0.02 | 0.05 |
| new | dataset_open | coverage=rebuilt | 8 | 0.34 | 0.26 |
| new | dataset_open | coverage=restored | 16 | 0.49 | 0.05 |
| new | dataset_stage | embed=fresh | 8 | 0.00 | 0.02 |
| new | text_sort | plain | 48 | 0.84 | 0.26 |
| old | dataset_load | embed=fresh | 4 | 0.00 | 0.04 |
| old | dataset_open | coverage=restored | 16 | 0.00 | 0.18 |
| old | dataset_stage | embed=cached | 4 | 0.02 | 0.41 |
| old | dataset_stage | embed=fresh | 4 | 0.00 | 0.02 |
| old | text_sort | plain | 48 | 0.85 | 0.28 |
| shipped | dataset_open | coverage=rebuilt | 8 | 0.15 | 0.70 |
| shipped | dataset_open | coverage=restored | 32 | 0.84 | 0.80 |
| shipped | dataset_stage | embed=cached | 4 | 0.85 | 0.61 |
| shipped | dataset_stage | embed=fresh | 12 | 0.67 | 0.98 |
| shipped | text_sort | plain | 96 | 0.80 | 0.53 |
