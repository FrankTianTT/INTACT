# target-domain-fast-adaptation

Target domain fast-adaptation for meta-RL with identifiable causal world model.

## Installation

```shell
conda create -n causal_meta python=3.11
conda activate causal_meta

# for cuda 11
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensordict-nightly -i https://pypi.org/simple
pip instiall -r requirements.txt
```

```shell
xvfb-run --auto-servernum --server-num=1 python examples/dreamers/train.py 
```