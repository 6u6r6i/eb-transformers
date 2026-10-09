#!/bin/bash
# Shared config for the eb-transformers setup/job scripts on NYU's Torch
# cluster (Greene was retired Spring 2026; this replaces the old Greene paths).
# Sourced by setup_env.sh and run_in_env.sh -- don't run this file directly.

# ENV_DIR / OVERLAY_NAME follow NYU HPC's standard Apptainer+overlay recipe
# for conda environments (https://services.rt.nyu.edu/docs/hpc/containers/singularity_with_conda/).
ENV_DIR="/scratch/${USER}/eb-transformers-env"
OVERLAY_NAME="overlay-15GB-500K.ext3"
OVERLAY_SRC="/share/apps/overlay-fs-ext3/${OVERLAY_NAME}.gz"

# Auto-pick the newest CUDA "devel" Apptainer/Singularity image NYU publishes,
# since the exact filename changes as NYU HPC updates these. If this comes up
# empty, log into Torch and run:
#   ls /share/apps/images/ | grep -i cuda
# then set SIF_IMAGE below by hand to one of those paths.
SIF_IMAGE="$(ls /share/apps/images/*cuda*devel*.sif 2>/dev/null | sort -V | tail -1)"

if [ -z "${SIF_IMAGE}" ] || [ ! -f "${SIF_IMAGE}" ]; then
    echo "ERROR: couldn't auto-detect a CUDA Singularity image on this cluster." >&2
    echo "Run: ls /share/apps/images/ | grep -i cuda" >&2
    echo "then hardcode SIF_IMAGE in hpc/config.sh to one of those paths." >&2
    exit 1
fi
