#!/bin/bash
# Seed-sharded evaluation, mirroring the --start/--end slicing eb_arena_mapper.py
# was designed for (see README.md: the paper's own runs used --seed 684
# --start 0 --end 4096 split across many jobs). Each array task handles one
# contiguous slice of seeds.
#
# Adjust TOTAL_SEEDS / --array to change how many tasks the sweep is split
# into, and --model to point at whichever checkpoint you're evaluating.
# Submit from the project root: sbatch hpc/eval_array.sbatch
#SBATCH --job-name=eb_arena
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --account=torch_pr_996_cds
#SBATCH --partition=a100_cds
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-9

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs eval_out

TOTAL_SEEDS=4096
NUM_TASKS=10   # must match the --array range above (0-9 = 10 tasks)
SEEDS_PER_TASK=$(( TOTAL_SEEDS / NUM_TASKS ))
START=$(( SLURM_ARRAY_TASK_ID * SEEDS_PER_TASK ))
END=$(( START + SEEDS_PER_TASK ))

bash hpc/run_in_env.sh python eb_arena_mapper.py \
    --model selected_models/T24r_24w.pkl \
    --seed 684 --start "${START}" --end "${END}" \
    --prior multinomial --theta_max 50 --seqlen 512 --batch 4000 \
    --llmap_out eval_out/
