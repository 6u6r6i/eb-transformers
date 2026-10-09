#!/bin/bash
# One-time setup of an Apptainer + Miniconda + PyTorch environment on Torch.
# Run on a LOGIN node (not inside a job), from the project root, e.g.:
#   ssh <netid>@login.torch.hpc.nyu.edu
#   cd ~/eb-transformers      # project code lives in $HOME, not $SCRATCH
#   bash hpc/setup_env.sh
#
# Before first run, sanity-check the paths hpc/config.sh assumes:
#   ls /share/apps/overlay-fs-ext3/
#   ls /share/apps/images/ | grep -i cuda
#
# Requires an active Slurm account/allocation (projects.hpc.nyu.edu) -- ask
# your PI to add you to a project first if `sbatch`/`srun` later refuse to run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

mkdir -p "${ENV_DIR}"
cd "${ENV_DIR}"

if [ ! -f "${OVERLAY_NAME}" ]; then
    echo "Copying + unpacking overlay image (one-time, ~15GB)..."
    cp "${OVERLAY_SRC}" "${OVERLAY_NAME}.gz"
    gunzip "${OVERLAY_NAME}.gz"
else
    echo "Overlay image already present at ${ENV_DIR}/${OVERLAY_NAME}, skipping copy."
fi

echo "Using image: ${SIF_IMAGE}"
echo "Installing Miniconda + Python deps inside the overlay (a few minutes)..."
# --fakeroot is required by Torch's Apptainer for :rw overlay mounts.
singularity exec --nv --fakeroot \
    --overlay "${ENV_DIR}/${OVERLAY_NAME}:rw" \
    "${SIF_IMAGE}" \
    bash "${SCRIPT_DIR}/container_install.sh"

echo "Done. Environment ready at ${ENV_DIR}/${OVERLAY_NAME}"
echo "Next: submit a job, e.g.  sbatch hpc/train.sbatch"
