# target-domain-fast-adaptation

<a href="https://github.com/FrankTianTT/causal-meta"><img src="https://github.com/FrankTianTT/causal-meta/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://github.com/FrankTianTT/causal-meta"><img src="https://codecov.io/github/FrankTianTT/causal-meta/branch/main/graph/badge.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/FrankTianTT/causal-meta/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
<a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white"></a>
<a href="https://www.python.org/downloads/release/python-311/"><img src="https://img.shields.io/badge/python-3.11-brightgreen"></a>

Target domain fast-adaptation for meta-RL with identifiable causal world model.

## Installation

```shell
conda create -n causal_meta python=3.11
conda activate causal_meta

# torch, torchvision, torchaudio, tensordict
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensordict-nightly==2023.11.26 -i https://pypi.org/simple

# rl
git clone https://github.com/FrankTianTT/rl.git -b dreamer_discount --depth 1
cd rl
python setup.py develop
cd ..

# causal-meta
https://github.com/FrankTianTT/causal-meta.git --depth 1
cd causal-meta
pip install -e .
```

```shell
xvfb-run --auto-servernum --server-num=1 python examples/dreamer/train.py
```
