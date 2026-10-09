#!/bin/bash
# Runs a command inside the eb-transformers Singularity+conda environment.
# Used by the sbatch scripts; can also be run directly from a login node for
# a quick interactive check (don't do real GPU work on a login node though).
#
# Usage: hpc/run_in_env.sh python eb_train.py --layers 12 ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# :ro here (vs :rw in setup_env.sh) -- multiple array-job tasks may read this
# overlay concurrently; only the one-time install should ever write to it.
singularity exec --nv \
    --overlay "${ENV_DIR}/${OVERLAY_NAME}:ro" \
    "${SIF_IMAGE}" \
    bash -c "source /ext3/env.sh && cd ${SLURM_SUBMIT_DIR:-$PWD} && $*"
