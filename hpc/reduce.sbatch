#!/bin/bash
# Aggregates all outputs from eval_array.sbatch into results.csv / tstats.csv.
# Run this AFTER every eval_array task has finished (check: squeue -u $USER).
# Submit from the project root: sbatch hpc/reduce.sbatch
#SBATCH --job-name=eb_reduce
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --account=torch_pr_996_cds

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs

bash hpc/run_in_env.sh python eb_arena_reducer.py eval_out/ eval_out/summary
