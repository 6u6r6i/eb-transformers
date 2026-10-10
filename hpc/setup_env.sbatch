#!/bin/bash
# One-time environment setup, run as a Slurm job instead of directly on a
# login node -- login nodes here kill `pip install torch ...` outright
# (torch's CUDA wheels run into the hundreds of MB each), presumably a
# memory/resource watchdog. Submit from the project root: sbatch hpc/setup_env.sbatch
#SBATCH --job-name=eb_env_setup
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --account=torch_pr_996_cds
# No GPU requested -- pip-installing packages doesn't need one, and asking
# for one just queues behind your group's GPU quota (QOSGrpGRES) for no
# reason. CPU-only partition queues near-instantly.
#SBATCH --partition=cpu_short

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs
bash hpc/setup_env.sh
