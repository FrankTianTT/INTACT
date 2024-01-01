from functools import partial
import os
import math
import argparse

import hydra
from omegaconf import OmegaConf
import torch
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tensordict.nn.probabilistic import InteractionType
from torchrl.envs import ParallelEnv, SerialEnv, TransformedEnv
from torchrl.record import VideoRecorder
from torchrl.modules.tensordict_module.exploration import AdditiveGaussianWrapper
from torchrl.objectives.dreamer import DreamerActorLoss, DreamerValueLoss
from torchrl.trainers.helpers.collectors import SyncDataCollector, MultiaSyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.trainers.trainers import Recorder

from causal_meta.helpers.envs import make_dreamer_env, create_make_env_list, build_make_env_list
from causal_meta.helpers.models import make_causal_dreamer
from causal_meta.helpers.logger import build_logger
from causal_meta.objectives.causal_dreamer import CausalDreamerModelLoss

from utils import grad_norm, match_length


def main(path, load_frame):
    cfg_path = os.path.join(path, ".hydra", "config.yaml")
    cfg = OmegaConf.load(cfg_path)

    if torch.cuda.is_available():
        device = torch.device(cfg.model_device)
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    make_env_fn = partial(
        make_dreamer_env,
        variable_num=cfg.variable_num,
        state_dim_per_variable=cfg.state_dim_per_variable,
        hidden_dim_per_variable=cfg.belief_dim_per_variable
    )
    train_oracle_context = torch.load(os.path.join(path, "train_oracle_context.pt"), map_location=device)
    test_oracle_context = torch.load(os.path.join(path, "test_oracle_context.pt"), map_location=device)
    train_make_env_list = build_make_env_list(cfg.env_name, make_env_fn, train_oracle_context)
    test_make_env_list = build_make_env_list(cfg.env_name, make_env_fn, test_oracle_context)

    task_num = len(train_make_env_list)
    proof_env = train_make_env_list[0]()
    world_model, model_based_env, actor_model, value_model, policy = make_causal_dreamer(
        cfg=cfg,
        proof_environment=proof_env,
        device=device,
    )

    if load_frame == -1:
        frames = sorted(map(int, os.listdir(os.path.join(path, "checkpoints"))))
        load_frame = frames[-1]
    world_model.load_state_dict(
        torch.load(os.path.join(path, "checkpoints", str(load_frame), f"world_model.pt"), map_location=device))
    actor_model.load_state_dict(
        torch.load(os.path.join(path, "checkpoints", str(load_frame), f"actor_model.pt"), map_location=device))
    value_model.load_state_dict(
        torch.load(os.path.join(path, "checkpoints", str(load_frame), f"value_model.pt"), map_location=device))

    state = torch.randn(30)
    belief = torch.randn(200)
    pixel = world_model.obs_decoder(state, belief)
    pixel = pixel.detach().permute(1, 2, 0).numpy()
    pixel = ((pixel + 0.5) * 255).astype("uint8")
    img = Image.fromarray(pixel)
    img.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--load_frame', type=int, default=-1)

    args = parser.parse_args()

    main(args.path, args.load_frame)
