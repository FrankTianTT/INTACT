from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.common import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import SafeModule
from torchrl.modules.models.model_based import ObsDecoder, ObsEncoder, RSSMPosterior
from torchrl.modules.models.models import MLP
from torchrl.trainers.helpers.models import _dreamer_make_mbenv, _dreamer_make_value_model, _dreamer_make_actors
from torchrl.modules.models.model_based import RSSMRollout
from torchrl.data.tensor_specs import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.modules import SafeProbabilisticTensorDictSequential, SafeProbabilisticModule
from tensordict.nn.probabilistic import InteractionType
from torchrl.modules.distributions import TanhNormal

from intact.modules.models.dreamer_world_model.causal_rssm_prior import CausalRSSMPrior
from intact.modules.models.dreamer_world_model.plain_rssm_prior import PlainRSSMPrior
from intact.modules.models.mdp_world_model import PlainMDPWorldModel, CausalWorldModel
from intact.modules.models.policy.actor import Actor
from intact.modules.models.policy.critic import Critic
from intact.modules.tensordict_module.dreamer_wrapper import DreamerWrapper
from intact.modules.tensordict_module.mdp_wrapper import MDPWrapper
from intact.envs.mdp_env import MDPEnv


def make_mdp_model(
    cfg: "DictConfig",  # noqa: F821
    proof_env: EnvBase,
    device: DEVICE_TYPING = "cpu",
):
    obs_dim = proof_env.observation_spec["observation"].shape[0]
    action_dim = proof_env.action_spec.shape[0]

    if cfg.model_type == "causal":
        wm_class = partial(CausalWorldModel, using_reinforce=cfg.using_reinforce, alpha=cfg.alpha)
    elif cfg.model_type == "plain":
        wm_class = PlainMDPWorldModel
    # elif cfg.model_type == "inn":
    #     wm_class = INNWorldModel
    else:
        raise NotImplementedError
    world_model = wm_class(
        obs_dim=obs_dim,
        action_dim=action_dim,
        meta=cfg.meta,
        max_context_dim=cfg.max_context_dim,
        task_num=cfg.task_num,
        hidden_dims=[cfg.hidden_size] * cfg.hidden_layers,
    )
    world_model = MDPWrapper(world_model).to(device)

    model_based_env = MDPEnv(world_model, termination_fns=cfg.termination_fns, reward_fns=cfg.reward_fns).to(device)
    model_based_env.set_specs_from_env(proof_env)

    return world_model, model_based_env


@dataclass
class MDPConfig:
    """MDP model config struct."""

    model_type = "causal"
    using_reinforce = True
    alpha = 1.0
    meta = False
    max_context_dim = 10
    task_num = 20
    hidden_size = 200
    hidden_layers = 2

    termination_fns = ""
    reward_fns = ""


def make_mdp_dreamer(
    cfg: "DictConfig",  # noqa: F821
    proof_env: EnvBase,
    device: DEVICE_TYPING = "cpu",
):
    obs_dim = proof_env.observation_spec["observation"].shape[0]
    action_dim = proof_env.action_spec.shape[0]

    world_model, model_based_env = make_mdp_model(cfg, proof_env, device=device)

    actor_module = Actor(
        action_dim=action_dim,
        state_or_obs_dim=obs_dim,
        context_dim=world_model.context_model.context_dim,
        is_mdp=True,
    ).to(device)
    actor_module.set_context_model(world_model.context_model)
    actor = SafeProbabilisticTensorDictSequential(
        SafeModule(
            actor_module,
            in_keys=["observation", "idx"],
            out_keys=["loc", "scale"],
            spec=CompositeSpec(
                **{
                    "loc": UnboundedContinuousTensorSpec(
                        proof_env.action_spec.shape,
                    ),
                    "scale": UnboundedContinuousTensorSpec(
                        proof_env.action_spec.shape,
                    ),
                }
            ),
        ),
        SafeProbabilisticModule(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={"tanh_loc": True},
            default_interaction_type=InteractionType.RANDOM,
            spec=CompositeSpec(**{"action": proof_env.action_spec.to("cpu")}),
        ),
    ).to(device)
    critic_module = Critic(
        state_or_obs_dim=obs_dim,
        context_dim=world_model.context_model.context_dim,
        is_mdp=True,
    ).to(device)
    critic_module.set_context_model(world_model.context_model)
    critic = SafeModule(
        critic_module,
        in_keys=["observation", "idx"],
        out_keys=["value"],
    ).to(device)

    return world_model, model_based_env, actor, critic
