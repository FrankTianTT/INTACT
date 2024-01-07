# target-domain-fast-adaptation

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