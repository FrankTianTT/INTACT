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
from torchrl.trainers.helpers.models import _dreamer_make_mbenv, _dreamer_make_value_model, \
    _dreamer_make_actors
from torchrl.modules.models.model_based import RSSMRollout

from causal_meta.modules.models.dreamer_world_model.causal_rssm_prior import CausalRSSMPrior
from causal_meta.modules.models.dreamer_world_model.plain_rssm_prior import PlainRSSMPrior
from causal_meta.modules.models.mdp_world_model import PlainMDPWorldModel, CausalWorldModel, INNWorldModel
from causal_meta.modules.tensordict_module.dreamer_wrapper import DreamerWrapper
from causal_meta.modules.tensordict_module.mdp_wrapper import MDPWrapper
from causal_meta.envs.mdp_env import MDPEnv


def make_mdp_model(
        cfg: "DictConfig",  # noqa: F821
        proof_env: EnvBase,
        device: DEVICE_TYPING = "cpu",
):
    obs_dim = proof_env.observation_spec["observation"].shape[0]
    action_dim = proof_env.action_spec.shape[0]

    if cfg.model_type == "causal":
        wm_class = partial(
            CausalWorldModel,
            reinforce=cfg.reinforce,
        )
    elif cfg.model_type == "plain":
        wm_class = PlainMDPWorldModel
    elif cfg.model_type == "inn":
        wm_class = INNWorldModel
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

    model_env = MDPEnv(
        world_model,
        termination_fns=cfg.termination_fns,
        reward_fns=cfg.reward_fns
    ).to(device)
    model_env.set_specs_from_env(proof_env)

    return world_model, model_env


def make_causal_dreamer(
        cfg: "DictConfig",  # noqa: F821
        proof_environment: EnvBase,
        device: DEVICE_TYPING = "cpu",
        action_key: str = "action",
        value_key: str = "state_value",
        use_decoder_in_env: bool = True,
):
    # Modules
    obs_encoder = ObsEncoder()
    obs_decoder = ObsDecoder()

    if cfg.model_type == "causal":
        rssm_prior_class = CausalRSSMPrior
    elif cfg.model_type == "plain":
        rssm_prior_class = PlainRSSMPrior
    else:
        raise NotImplementedError

    rssm_prior = rssm_prior_class(
        action_dim=proof_environment.action_space.shape[0],
        variable_num=cfg.variable_num,
        state_dim_per_variable=cfg.state_dim_per_variable,
        hidden_dim_per_variable=cfg.hidden_dim_per_variable,
        rnn_input_dim_per_variable=cfg.rnn_input_dim_per_variable,
        max_context_dim=cfg.max_context_dim,
        task_num=cfg.task_num,
        residual=cfg.residual,
    )

    rssm_posterior = RSSMPosterior(
        hidden_dim=cfg.hidden_dim_per_variable * cfg.variable_num,
        state_dim=cfg.state_dim_per_variable * cfg.variable_num,
    )
    reward_module = MLP(
        out_features=1, depth=2, num_cells=cfg.hidden_size, activation_class=nn.ELU
    )

    if cfg.pred_continue:
        continue_module = MLP(
            out_features=1, depth=2, num_cells=cfg.hidden_size, activation_class=nn.ELU
        )
    else:
        continue_module = None

    world_model = _dreamer_make_world_model(
        obs_encoder, obs_decoder, rssm_prior, rssm_posterior, reward_module, continue_module
    ).to(device)
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        tensordict = proof_environment.fake_tensordict().unsqueeze(-1)
        tensordict = tensordict.to_tensordict().to(device)
        tensordict = tensordict.to(device)
        world_model(tensordict)

    model_based_env = _dreamer_make_mbenv(
        reward_module,
        continue_module,
        rssm_prior,
        obs_decoder,
        proof_environment,
        use_decoder_in_env,
        state_dim=cfg.state_dim_per_variable * cfg.variable_num,
        rssm_hidden_dim=cfg.hidden_dim_per_variable * cfg.variable_num,
    )
    model_based_env = model_based_env.to(device)

    actor_simulator, actor_realworld = _dreamer_make_actors(
        obs_encoder,
        rssm_prior,
        rssm_posterior,
        cfg.hidden_size,
        action_key,
        proof_environment,
    )
    actor_simulator = actor_simulator.to(device)

    value_model = _dreamer_make_value_model(cfg.hidden_size, value_key)
    value_model = value_model.to(device)
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        tensordict = model_based_env.fake_tensordict().unsqueeze(-1)
        tensordict = tensordict.to(device)
        tensordict = actor_simulator(tensordict)
        value_model(tensordict)

    actor_realworld = actor_realworld.to(device)

    del tensordict
    return world_model, model_based_env, actor_simulator, value_model, actor_realworld


def _dreamer_make_world_model(
        obs_encoder, obs_decoder, rssm_prior, rssm_posterior, reward_module, continue_module
):
    # World Model and reward model
    rssm_rollout = RSSMRollout(
        SafeModule(
            rssm_prior,
            in_keys=["state", "belief", "action"],
            out_keys=[
                ("next", "prior_mean"),
                ("next", "prior_std"),
                "_",
                ("next", "belief"),
                "causal_mask",
            ],
        ),
        SafeModule(
            rssm_posterior,
            in_keys=[("next", "belief"), ("next", "encoded_latents")],
            out_keys=[
                ("next", "posterior_mean"),
                ("next", "posterior_std"),
                ("next", "state"),
            ],
        ),
    )

    obs_encoder = SafeModule(
        obs_encoder,
        in_keys=[("next", "pixels")],
        out_keys=[("next", "encoded_latents")],
    )

    obs_decoder = SafeModule(
        obs_decoder,
        in_keys=[("next", "state"), ("next", "belief")],
        out_keys=[("next", "reco_pixels")],
    )

    reward_model = SafeModule(
        reward_module,
        in_keys=[("next", "state"), ("next", "belief")],
        out_keys=[("next", "reward")],
    )
    if continue_module is not None:
        continue_model = SafeModule(
            continue_module,
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=[("next", "pred_continue")],
        )
    else:
        continue_model = None

    world_model = DreamerWrapper(
        obs_encoder=obs_encoder,
        rssm_rollout=rssm_rollout,
        obs_decoder=obs_decoder,
        reward_model=reward_model,
        continue_model=continue_model,
    )
    return world_model


@dataclass
class DreamerConfig:
    """Dreamer model config struct."""

    batch_length: int = 50

    variable_num: int = 10
    state_dim_per_variable: int = 3
    hidden_dim_per_variable: int = 20
    rnn_input_dim_per_variable: int = 20
    residual: bool = False
    logits_clip: float = 3.0
    max_context_dim: int = 0
    task_num: int = 0

    mlp_num_units: int = 400
    grad_clip: int = 100
    world_model_lr: float = 6e-4
    actor_value_lr: float = 8e-5
    context_lr: float = 1e-1
    mask_logits_lr: float = 1e-3
    lambda_kl: float = 1.0
    lambda_reco: float = 1.0
    lambda_reward: float = 1.0
    lambda_continue: float = 1.0
    imagination_horizon: int = 15
    model_device: str = ""
    # Decay of the reward moving averaging
    exploration: str = "additive_gaussian"
    # One of "additive_gaussian", "ou_exploration" or ""
    discount_loss: bool = True
    # Whether to use the discount loss
    pred_continue: bool = True
    # Whether to predict the continue signal
    train_agent_frames: int = 100000

    train_causal_iters: int = 10
    train_model_iters: int = 50

    sparse_weight: float = 0.02
    context_sparse_weight: float = 0.01
    context_max_weight: float = 0.2
    sampling_times: int = 30


def test_make_causal_dreamer():
    from torchrl.envs import GymEnv, TransformedEnv, Compose, ToTensorImage, DoubleToFloat, TensorDictPrimer
    from torchrl.data import UnboundedContinuousTensorSpec

    cfg = DreamerConfig()

    env = TransformedEnv(
        GymEnv("CartPoleContinuous-v0", from_pixels=True),
        Compose(ToTensorImage(), DoubleToFloat())
    )
    default_dict = {
        "state": UnboundedContinuousTensorSpec(shape=torch.Size((
            *env.batch_size, cfg.variable_num * cfg.state_dim_per_variable
        ))),
        "belief": UnboundedContinuousTensorSpec(shape=torch.Size((
            *env.batch_size, cfg.variable_num * cfg.hidden_dim_per_variable
        )))
    }
    env.append_transform(TensorDictPrimer(random=False, default_value=0, **default_dict))

    world_model, model_based_env, actor_simulator, value_model, actor_realworld = make_causal_dreamer(cfg, env)

    world_model.get_parameter("transition_model.0.weight")


if __name__ == '__main__':
    # test_make_causal_dreamer()

    import matplotlib.pyplot as plt

    x = torch.randn(1000)
    plt.hist(x, bins=100)
    plt.show()

    x = torch.pow(x, 3)
    plt.hist(x, bins=100)
    plt.show()
