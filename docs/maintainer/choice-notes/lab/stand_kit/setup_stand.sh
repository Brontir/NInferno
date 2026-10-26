#!/bin/bash
# A new stand for Claude's sessions: the build tools and libraries the other stands have (no driver,
# no CUDA change), the test venv, /root/claude with the helper scripts, and a clone to work from.
set -u
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq cmake ninja-build ccache pkg-config patchelf libcurl4-openssl-dev libssl-dev \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev > /root/setup_apt.log 2>&1
echo "apt rc=$?"; for c in cmake ninja ccache pkg-config; do printf "%s " $(command -v $c || echo MISSING-$c); done; echo
mkdir -p /root/claude /root/claude/logs /root/claude/mix
cd /root/claude
python3 -m venv venv-torch && venv-torch/bin/pip install -q --upgrade pip > /dev/null 2>&1
venv-torch/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch > /root/setup_pip.log 2>&1
venv-torch/bin/pip install -q -r /root/claude/venv_freeze_nontorch.txt >> /root/setup_pip.log 2>&1
echo "pip rc=$?"; venv-torch/bin/python -c "import torch, numpy, gguf, safetensors; print('venv ok', torch.__version__)"
[ -d wt_tst ] || git clone -q https://github.com/Brontir/NInferno wt_tst
git -C wt_tst log --oneline -1
echo SETUPDONE
