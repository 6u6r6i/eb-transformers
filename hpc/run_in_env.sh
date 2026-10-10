#!/bin/bash
# Runs a command inside the eb-transformers Singularity+conda environment.
# Used by the sbatch scripts; can also be run directly from a login node for
# a quick interactive check (don't do real GPU work on a login node though).
#
# Usage: hpc/run_in_env.sh python eb_train.py --layers 12 ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# --bind "$HOME" and "/scratch" because this site's Apptainer doesn't
# auto-bind either (see setup_env.sh / config.sh); the project (and
# $SLURM_SUBMIT_DIR) lives under $HOME, the conda env under $SCRATCH.
singularity exec --nv \
    --bind "${HOME}:${HOME}" \
    --bind "/scratch:/scratch" \
    "${SIF_IMAGE}" \
    bash -c "source ${ENV_DIR}/env.sh && cd ${SLURM_SUBMIT_DIR:-$PWD} && export PYTHONPATH=\"\$(pwd)/src:\${PYTHONPATH:-}\" && $*"
