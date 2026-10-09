#!/bin/bash
# Runs INSIDE the Singularity container -- called by setup_env.sh, not by hand.
# Installs Miniconda into the writable overlay at /ext3, plus everything
# eb-transformers needs, and writes /ext3/env.sh so job scripts can later just
# `source /ext3/env.sh` to get the environment back.
set -euo pipefail

if [ ! -d /ext3/miniconda3 ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /ext3/miniconda3
    rm /tmp/miniconda.sh
fi

cat > /ext3/env.sh <<'EOF'
#!/bin/bash
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONNOUSERSITE=1
EOF

source /ext3/env.sh
pip install --upgrade pip

# torch: installs whatever CUDA build pip's default index resolves for this
# image's CUDA toolkit. If `torch.cuda.is_available()` comes back False once
# a job runs on a GPU node, check https://pytorch.org for the --index-url
# matching this container's CUDA version and reinstall torch with that.
pip install torch numpy tqdm tabulate pandas scipy scikit-learn seaborn matplotlib \
            nltk yellowbrick torcheval easydict

python -c "import nltk; nltk.download('stopwords')"

echo "Container-side install complete."
