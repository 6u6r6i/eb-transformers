#!/bin/bash
# One-time setup of an Apptainer + Miniconda + PyTorch environment on Torch.
# Run on a LOGIN node (not inside a job), from the project root, e.g.:
#   ssh <netid>@login.torch.hpc.nyu.edu
#   cd ~/eb-transformers      # project code lives in $HOME, not $SCRATCH
#   bash hpc/setup_env.sh
#
# Before first run, sanity-check the path hpc/config.sh assumes:
#   ls /share/apps/images/ | grep -i cuda
#
# Requires an active Slurm account/allocation (projects.hpc.nyu.edu) -- ask
# your PI to add you to a project first if `sbatch`/`srun` later refuse to run.
#
# No --fakeroot / --overlay here -- see hpc/config.sh for why: that stack was
# measured to be drastically slow for this many small files on this cluster.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

mkdir -p "${ENV_DIR}"

echo "Using image: ${SIF_IMAGE}"
echo "Installing Miniconda + Python deps on \$SCRATCH (a few minutes)..."
# --bind "$HOME" and "/scratch" are required because this site's Apptainer
# does not auto-bind either into the container by default.
singularity exec --nv \
    --bind "${HOME}:${HOME}" \
    --bind "/scratch:/scratch" \
    --env "ENV_DIR=${ENV_DIR}" \
    "${SIF_IMAGE}" \
    bash "${SCRIPT_DIR}/container_install.sh"

echo "Done. Environment ready at ${ENV_DIR}/miniconda3"
echo "Next: submit a job, e.g.  sbatch hpc/train.sbatch"
