#!/bin/bash
# Single-GPU training job. Submit from the project root: sbatch hpc/train.sbatch
#
# --account/--partition below match `sacctmgr -p show assoc user=$USER` and
# `sinfo -o "%P %G"` run 2026-10 -- re-check if these ever stop working:
#   sacctmgr -p show assoc user=$USER format=account,user,partition
#   sinfo -o "%P %G"
# a100 is already overkill for this model (dmodel=32); swap to rtx6000_cds /
# l40s_cds / h200_cds etc. only if a100_cds is ever saturated.
#SBATCH --job-name=eb_train
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --account=torch_pr_996_cds
#SBATCH --partition=a100_cds
#SBATCH --gres=gpu:a100:1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs

# The (k, n) Good-Turing MoE experiment: k fixed via --seqlen, n via
# --n_samples, MoE enabled via --moe_experts.
bash hpc/run_in_env.sh python eb_train.py \
    --layers 24 --heads 4 --dmodel 32 \
    --prior goodturing --seqlen 10000 --n_samples 20000 --gt_alpha 0.3 \
    --channel poisson --moe_experts 8 --moe_aux_loss_coef 0.01 \
    --train_steps 100000 \
    --save_dir "/scratch/${USER}/eb-transformers-runs"

# Plain dense-MLP baseline on the standard neural prior (mirrors eb_train.py's
# own argparse defaults) -- swap back to this to compare against the above:
# bash hpc/run_in_env.sh python eb_train.py \
#     --layers 24 --heads 4 --dmodel 32 \
#     --prior neural --channel poisson \
#     --train_steps 100000 \
#     --save_dir "/scratch/${USER}/eb-transformers-runs"
