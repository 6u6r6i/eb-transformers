#!/bin/bash
# Shared config for the eb-transformers setup/job scripts on NYU's Torch
# cluster (Greene was retired Spring 2026; this replaces the old Greene paths).
# Sourced by setup_env.sh and run_in_env.sh -- don't run this file directly.

# NYU's documented recipe (https://services.rt.nyu.edu/docs/hpc/containers/singularity_with_conda/)
# puts the conda env inside an Apptainer --fakeroot overlay image, to avoid
# burning through $SCRATCH's inode quota if many users each did this.
# On-cluster testing (2026-10-10) found that --fakeroot + fuse-overlayfs +
# squashfuse is drastically slow for installs with many small files (a
# Miniconda install's `rm -rf` alone ran 5+ minutes at ~0% CPU) -- plausibly
# what actually caused an earlier run to sit for ~20 hours, not a true hang.
# We install straight onto $SCRATCH instead: no --fakeroot, no overlay, real
# filesystem I/O. One user's one conda env (tens of thousands of files) is
# nowhere near $SCRATCH's 5,000,000-inode quota, so this trade is fine here.
ENV_DIR="/scratch/${USER}/eb-transformers-env"

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
