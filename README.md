# INTACT

<a href="https://github.com/FrankTianTT/causal-meta"><img src="https://github.com/FrankTianTT/causal-meta/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://app.codecov.io/github/FrankTianTT/intact"><img src="https://codecov.io/github/FrankTianTT/intact/branch/main/graph/badge.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/FrankTianTT/causal-meta/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
<a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white"></a>
<a href="https://www.python.org/downloads/release/python-311/"><img src="https://img.shields.io/badge/python-3.11-brightgreen"></a>

INTACT stands for *Identifiable aNd TrAnsferable Causal meTa World Model*. This framework is introduced to address limitations by enabling the acquisition of fully identifiable contextual representations under relatively mild conditions, while promoting rapid adaptation to new contexts without compromising the original semantics.

## Installation

Using python 3.11 for example:
```shell
# create conda env
conda create -n intact python=3.11
conda activate intact

# install torch, torchvision, tensordict
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tensordict-nightly -i https://pypi.org/simple # do not use mirror

# install intact
https://github.com/FrankTianTT/intact.git --depth 1
cd intact
pip install -e .
```

## Using

Dreamer in POMDP:
```shell
xvfb-run --auto-servernum --server-num=1 python examples/dreamer/train.py
```

Dreamer in MDP:
```shell
python examples/dreamer_mdp/train.py
```

MPC in MDP:
```shell
python examples/mdp/train.py
```

## Design

![Design](images/model-uml.png)
