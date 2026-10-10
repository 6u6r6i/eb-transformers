#!/bin/bash
# Runs INSIDE the Apptainer container -- called by setup_env.sh, not by hand.
# Installs Miniconda directly onto $SCRATCH (via $ENV_DIR, passed in from the
# host through --env), plus everything eb-transformers needs, and writes
# $ENV_DIR/env.sh so job scripts can later just `source` it to get the
# environment back.
set -euo pipefail

MINICONDA_DIR="${ENV_DIR}/miniconda3"

# Check for the actual marker file, not just the directory -- an interrupted
# install (killed mid-extraction, e.g. a dropped SSH session) can leave a
# miniconda3 directory that exists but is incomplete/broken, which a plain
# `[ -d ... ]` check would wrongly treat as "already installed".
if [ ! -f "${MINICONDA_DIR}/etc/profile.d/conda.sh" ]; then
    rm -rf "${MINICONDA_DIR}"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${MINICONDA_DIR}"
    rm /tmp/miniconda.sh
fi

cat > "${ENV_DIR}/env.sh" <<EOF
#!/bin/bash
source ${MINICONDA_DIR}/etc/profile.d/conda.sh
conda activate base
export PYTHONNOUSERSITE=1
EOF

source "${ENV_DIR}/env.sh"
pip install --upgrade pip

# torch: installs whatever CUDA build pip's default index resolves for this
# image's CUDA toolkit. If `torch.cuda.is_available()` comes back False once
# a job runs on a GPU node, check https://pytorch.org for the --index-url
# matching this container's CUDA version and reinstall torch with that.
pip install torch numpy tqdm tabulate pandas scipy scikit-learn seaborn matplotlib \
            nltk yellowbrick torcheval easydict

python -c "import nltk; nltk.download('stopwords')"

echo "Container-side install complete."
