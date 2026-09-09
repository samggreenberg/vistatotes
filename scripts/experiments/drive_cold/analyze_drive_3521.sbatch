#!/bin/bash
# Analysis + figures for #3521, chained off the sweep with --dependency=afterok.
#
# On a GPU node on purpose: `load_step_weights` reads the device from the
# PROCESS, not from an argument, so the `shipped` arm silently becomes a CPU row
# on a login node and every number in the table would describe hardware the
# sweep did not run on (#3345).
#SBATCH --job-name=drive3521a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:45:00
#SBATCH --output=/expscratch/sgreenberg/drive-3521/slurm-analyze-%j.out

set -euo pipefail
EXP=/expscratch/sgreenberg/drive-3521
WT=/exp/sgreenberg/projects/vts-drive-3521

module load python/3.12.3
source /exp/sgreenberg/projects/VTSearch/.venv/bin/activate
cd "$WT"

python scripts/experiments/drive_cold/analyze_drive_3521.py --exp "$EXP" 2>&1 | tee "$EXP/analyze.log"
python scripts/experiments/drive_cold/figures_drive_3521.py --exp "$EXP" 2>&1 | tee -a "$EXP/analyze.log"
touch "$EXP/DONE.analyze"
