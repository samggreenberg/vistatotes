# MLP vs SVM — the definitive experiment (runner)

Code that runs the MLP-vs-SVM ranker study on the HLTCOE Grid and generates the
report. The design, the pre-registered decision rule and the verdict are in
[`docs/experiments/2026-07-22-mlp-vs-svm/REPORT.md`](../../../docs/experiments/2026-07-22-mlp-vs-svm/REPORT.md). Everything is image + SigLIP only.

## What each stage does

| Stage | Script | Output |
|---|---|---|
| 0 · prepare | `prepare_data.py <ids>` | Embeds each dataset once with SigLIP, caching the pickle under `$VTSEARCH_DATA_DIR/embeddings`; writes `prepare_info.json` (per-category counts). |
| A · screen | `stage_a_screen.py` | `stage_a.csv` — label-count sweep over the widened SVM grid (linear/rbf/poly/sigmoid × C × gamma) to pick the best config per kernel family. |
| B · definitive | `stage_b_autopilot.py` | `stage_b/task_<i>.csv` — one SLURM-array task per `(dataset, category, prevalence_arm, seed)` cell, all trainers inside. The Autopilot voting simulation with production calibration; per-step FPR/FNR/cost/AUROC/AP + timing. |
| C · timing | `stage_c_timing.py` | `stage_c.csv` + `stage_c_parity.json` — GPU train/inference scaling (torch MLP vs cuML SVM, median-of-7). |
| report | `summarize.py` | `REPORT.md` + figures (deterministic from the CSVs). |

`common.py` sets the experiment data/model/HF dirs and neutralises the main venv's
editable-install finder so `import vtscore` resolves to *this* worktree.
`experiment_config.py` holds the pre-registered grid (env-overridable knobs).

## Run it

```bash
# One-shot dependency chain (submits prepare -> A + B-array + C -> summarize):
bash queue_all.sh 288        # 288 = safe upper bound on array cells

# Or by hand, sized exactly:
sbatch ... --wrap "source ../../../gridenv.sh && cd $PWD && python prepare_data.py caltech101_m caltech256_a visual_genome_m"
N=$(python stage_b_autopilot.py --print-cells)      # after prepare
sbatch --array=0-$((N-1))%24 ... --wrap "... python stage_b_autopilot.py"
```

## Sizing knobs (env vars, read by `experiment_config.py`)

| Var | Default | Meaning |
|---|---|---|
| `MLPSVM_DATASETS` | `caltech101_m,caltech256_a,visual_genome_m` | Datasets in the grid |
| `MLPSVM_N_CATEGORIES` | `6` | Categories per dataset (spanning common→rare) |
| `MLPSVM_N_SEEDS` | `8` | Seeds (paired across trainers) |
| `MLPSVM_MAX_STEPS` | `200` | Vote budget per trajectory |
| `MLPSVM_RARE_PREVALENCE` | `0.01` | Rare-arm target prevalence |
| `MLPSVM_TRAINERS` | `app,svm_linear,svm_rbf` | Trainers in the definitive run |

## Notes / gotchas

- **cuML SVM is present but broken on this cluster** (nvrtc compiles cu13 fp4
  headers under a cu12 toolchain). `train_svm(backend="auto")` degrades to
  sklearn-CPU and labels rows `sklearn-cpu`; the MLP still uses torch-CUDA. Stage
  C therefore compares MLP-GPU vs SVM-CPU, and the report says so. To force CPU
  everywhere set `VTSEARCH_DISABLE_CUML=1`.
- Rare-arm cells for low-prevalence categories are skipped (they'd leave < 15
  positives); those tasks write a 0-row CSV, which `summarize.py` drops.
- Results live under `/exp/$USER/mlp-svm/results`.
