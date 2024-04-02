xvfb-run --auto-servernum --server-num=1  python examples/mpc/train.py overrides=half_cheetah meta=False model_device=cuda:0 init_frames_per_task=0


mbrl:
```bash
python -m mbrl.examples.main algorithm=pets overrides=pets_halfcheetah device="cuda:0" overrides.learned_rewards=True overrides.obs_process_fn=null dynamics_model=gaussian_mlp
```
