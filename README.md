# target-domain-fast-adaptation

Target domain fast-adaptation for meta-RL with identifiable causal world model.

## Installation

```shell
conda create -n tdfa python=3.8
conda activate tdfa
conda install pytorch::pytorch -c pytorch

pip install tensordict-nightly -i https://pypi.org/simple
pip instiall -r requirements.txt
```

```shell
xvfb-run --auto-servernum --server-num=1 python example/dreamers/dreamer.py 
```