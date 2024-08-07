from dataclasses import dataclass
from functools import partial
from warnings import catch_warnings, filterwarnings

import torch
from omegaconf import DictConfig
from tensordict.nn.probabilistic import set_interaction_type, InteractionType
from torchrl.envs import TransformedEnv, SerialEnv
from torchrl.envs.utils import step_mdp
from torchrl.record import VideoRecorder
from tqdm import tqdm

from intact.utils.envs import build_make_env_list, make_mdp_env


def evaluate_policy(
    cfg: DictConfig,
    oracle_context,
    policy,
    logger=None,
    log_idx="",
    make_env_fn=make_mdp_env,
    log_prefix="meta_train",
    disable_pixel_if_possible=True,
):
    """Evaluate policy.

    Args:
        cfg: Configuration.
        oracle_context: Oracle context.
        policy: Policy.
        logger: Logger.
        log_idx: Log index.
        make_env_fn: Make environment function.
        log_prefix: Log prefix.
        disable_pixel_if_possible: Disable pixel if possible.

    """
    if hasattr(policy, "parameters"):
        device = next(policy.parameters()).device
    else:
        device = "cpu"

    pbar = tqdm(
        total=cfg.eval_repeat_nums * cfg.env_max_steps,
        desc="{}_eval".format(log_prefix),
    )
    repeat_rewards = []
    repeat_lengths = []
    for repeat in range(cfg.eval_repeat_nums):
        if disable_pixel_if_possible:
            make_env_fn = partial(make_env_fn, pixel=repeat < cfg.eval_record_nums)
        make_env_list = build_make_env_list(cfg.env_name, make_env_fn, oracle_context)
        eval_env = SerialEnv(len(make_env_list), make_env_list, shared_memory=False)
        if repeat < cfg.eval_record_nums:
            eval_env = TransformedEnv(eval_env, VideoRecorder(logger, log_prefix))

        rewards = torch.zeros(len(make_env_list))
        lengths = torch.zeros(len(make_env_list))
        tensordict = eval_env.reset().to(device)
        ever_done = torch.zeros(*tensordict.batch_size, 1).to(bool)

        for _ in range(cfg.env_max_steps):
            pbar.update()
            with set_interaction_type(InteractionType.MODE):
                if disable_pixel_if_possible and "pixels" in tensordict.keys():
                    del tensordict["pixels"]
                action = policy(tensordict.to(device)).cpu()
                with catch_warnings():
                    filterwarnings("ignore", category=UserWarning)
                    tensordict = eval_env.step(action)

            reward = tensordict.get(("next", "reward"))
            reward[ever_done] = 0
            rewards += reward.reshape(-1)
            ever_done |= tensordict.get(("next", "done"))
            lengths += (~ever_done).float().reshape(-1)
            if ever_done.all():
                break
            else:
                tensordict = step_mdp(tensordict, exclude_action=False)
        print(rewards)
        repeat_rewards.append(rewards)
        repeat_lengths.append(lengths)

        if repeat < cfg.eval_record_nums:
            eval_env.transform.dump(suffix=str(log_idx))

    if logger is not None:
        logger.add_scaler(
            "{}/eval_episode_reward".format(log_prefix),
            torch.stack(repeat_rewards).mean(),
        )
        logger.add_scaler(
            "{}/eval_episode_length".format(log_prefix),
            torch.stack(repeat_lengths).mean(),
        )
        logger.dump_scaler(log_idx)

    return torch.stack(repeat_rewards)


@dataclass
class EvaluateConfig:
    env_name = "MyCartPole-v0"

    eval_repeat_nums = 1
    eval_record_nums = 0

    env_max_steps = 200
