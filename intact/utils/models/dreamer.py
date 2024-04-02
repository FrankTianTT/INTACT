from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.common import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import SafeModule
from torchrl.modules.models.model_based import ObsDecoder, ObsEncoder, RSSMPosterior, RSSMRollout
from torchrl.modules.models.models import MLP
from torchrl.trainers.helpers.models import _dreamer_make_mbenv
from torchrl.data.tensor_specs import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.modules import SafeProbabilisticTensorDictSequential, SafeProbabilisticModule, SafeSequential
from tensordict.nn.probabilistic import InteractionType
from torchrl.modules.distributions import TanhNormal, TruncatedNormal
from torchrl.envs.transforms import TensorDictPrimer, TransformedEnv
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.modules.tensordict_module.world_models import DreamerWrapper as OriginalDreamerWrapper

from intact.modules.models.dreamer_world_model.causal_rssm_prior import CausalRSSMPrior
from intact.modules.models.dreamer_world_model.plain_rssm_prior import PlainRSSMPrior
from intact.modules.models.policy.actor import Actor
from intact.modules.models.policy.critic import Critic
from intact.modules.tensordict_module.dreamer_wrapper import DreamerWrapper


def make_dreamer(
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
        rssm_prior_class = partial(
            CausalRSSMPrior,
            using_cross_belief=cfg.using_cross_belief,
            using_reinforce=cfg.using_reinforce,
        )
    elif cfg.model_type == "plain":
        rssm_prior_class = PlainRSSMPrior
    else:
        raise NotImplementedError

    rssm_prior = rssm_prior_class(
        action_dim=proof_environment.action_space.shape[0],
        variable_num=cfg.variable_num,
        state_dim_per_variable=cfg.state_dim_per_variable,
        belief_dim_per_variable=cfg.belief_dim_per_variable,
        disable_belief=cfg.disable_belief,
        meta=cfg.meta,
        max_context_dim=cfg.max_context_dim,
        task_num=cfg.task_num,
        residual=cfg.residual,
    )

    rssm_posterior = RSSMPosterior(
        hidden_dim=cfg.belief_dim_per_variable * cfg.variable_num,
        state_dim=cfg.state_dim_per_variable * cfg.variable_num,
    )
    reward_module = MLP(out_features=1, depth=2, num_cells=cfg.hidden_size, activation_class=nn.ELU)

    if cfg.pred_continue:
        continue_module = MLP(out_features=1, depth=2, num_cells=cfg.hidden_size, activation_class=nn.ELU)
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
        rssm_hidden_dim=cfg.belief_dim_per_variable * cfg.variable_num,
    )
    model_based_env = model_based_env.to(device)

    actor_simulator, actor_realworld = _dreamer_make_actors(
        state_dim=cfg.state_dim_per_variable * cfg.variable_num,
        belief_dim=cfg.belief_dim_per_variable * cfg.variable_num,
        context_model=rssm_prior.context_model,
        obs_encoder=obs_encoder,
        rssm_prior=rssm_prior,
        rssm_posterior=rssm_posterior,
        mlp_num_units=cfg.hidden_size,
        action_key=action_key,
        proof_environment=proof_environment,
        actor_dist_type=cfg.actor_dist_type,
    )
    actor_simulator = actor_simulator.to(device)

    value_model = _dreamer_make_value_model(
        state_dim=cfg.state_dim_per_variable * cfg.variable_num,
        belief_dim=cfg.belief_dim_per_variable * cfg.variable_num,
        context_model=rssm_prior.context_model,
        mlp_num_units=cfg.hidden_size,
        value_key=value_key,
    )
    value_model = value_model.to(device)
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        tensordict = model_based_env.fake_tensordict().unsqueeze(-1)
        tensordict = tensordict.to(device)
        tensordict = actor_simulator(tensordict)
        value_model(tensordict)

    actor_realworld = actor_realworld.to(device)

    del tensordict
    return world_model, model_based_env, actor_simulator, value_model, actor_realworld


def _dreamer_make_world_model(obs_encoder, obs_decoder, rssm_prior, rssm_posterior, reward_module, continue_module):
    # World Model and reward model
    rssm_rollout = RSSMRollout(
        SafeModule(
            rssm_prior,
            in_keys=["state", "belief", "action", "idx"],
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


def _dreamer_make_mbenv(
    reward_module,
    continue_module,
    rssm_prior,
    obs_decoder,
    proof_environment,
    use_decoder_in_env,
    state_dim,
    rssm_hidden_dim,
):
    # MB environment
    if use_decoder_in_env:
        mb_env_obs_decoder = SafeModule(
            obs_decoder,
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=[("next", "reco_pixels")],
        )
    else:
        mb_env_obs_decoder = None

    transition_model = SafeSequential(
        SafeModule(
            rssm_prior,
            in_keys=["state", "belief", "action", "idx"],
            out_keys=[
                "_",
                "_",
                "state",
                "belief",
            ],
        ),
    )
    reward_model = SafeModule(
        reward_module,
        in_keys=["state", "belief"],
        out_keys=["reward"],
    )
    if continue_module is not None:
        continue_model = SafeModule(
            continue_module,
            in_keys=["state", "belief"],
            out_keys=["pred_continue"],
        )
    else:
        continue_model = None

    model_based_env = DreamerEnv(
        world_model=OriginalDreamerWrapper(transition_model, reward_model, continue_model),
        prior_shape=torch.Size([state_dim]),
        belief_shape=torch.Size([rssm_hidden_dim]),
        obs_decoder=mb_env_obs_decoder,
    )

    model_based_env.set_specs_from_env(proof_environment)
    model_based_env = TransformedEnv(model_based_env)
    default_dict = {
        "state": UnboundedContinuousTensorSpec(state_dim),
        "belief": UnboundedContinuousTensorSpec(rssm_hidden_dim),
        # "action": proof_environment.action_spec,
    }
    model_based_env.append_transform(TensorDictPrimer(random=False, default_value=0, **default_dict))
    return model_based_env


def _dreamer_make_actors(
    state_dim,
    belief_dim,
    context_model,
    obs_encoder,
    rssm_prior,
    rssm_posterior,
    mlp_num_units,
    action_key,
    proof_environment,
    actor_dist_type="truncated_normal",
):
    actor_module = Actor(
        state_or_obs_dim=state_dim,
        belief_dim=belief_dim,
        action_dim=proof_environment.action_spec.shape[0],
        context_dim=context_model.context_dim,
        depth=3,
        num_cells=mlp_num_units,
        activation_class=nn.ELU,
        is_mdp=False,
    )
    actor_module.set_context_model(context_model)

    actor_simulator = _dreamer_make_actor_sim(action_key, proof_environment, actor_module, actor_dist_type)
    actor_realworld = _dreamer_make_actor_real(
        obs_encoder, rssm_prior, rssm_posterior, actor_module, action_key, proof_environment, actor_dist_type
    )
    return actor_simulator, actor_realworld


def _dreamer_make_actor_sim(action_key, proof_environment, actor_module, actor_dist_type):
    distribution_class = {"truncated_normal": TruncatedNormal, "tanh_normal": TanhNormal}[actor_dist_type]
    actor_simulator = SafeProbabilisticTensorDictSequential(
        SafeModule(
            actor_module,
            in_keys=["state", "idx", "belief"],
            out_keys=["loc", "scale"],
            spec=CompositeSpec(
                **{
                    "loc": UnboundedContinuousTensorSpec(
                        proof_environment.action_spec.shape,
                        device=proof_environment.action_spec.device,
                    ),
                    "scale": UnboundedContinuousTensorSpec(
                        proof_environment.action_spec.shape,
                        device=proof_environment.action_spec.device,
                    ),
                }
            ),
        ),
        SafeProbabilisticModule(
            in_keys=["loc", "scale"],
            out_keys=[action_key],
            default_interaction_type=InteractionType.RANDOM,
            distribution_class=distribution_class,
            distribution_kwargs={"tanh_loc": True},
            spec=CompositeSpec(**{action_key: proof_environment.action_spec}),
        ),
    )
    return actor_simulator


def _dreamer_make_actor_real(
    obs_encoder, rssm_prior, rssm_posterior, actor_module, action_key, proof_environment, actor_dist_type
):
    distribution_class = {"truncated_normal": TruncatedNormal, "tanh_normal": TanhNormal}[actor_dist_type]
    # actor for real world: interacts with states ~ posterior
    # Out actor differs from the original paper where first they compute prior and posterior and then act on it
    # but we found that this approach worked better.
    actor_realworld = SafeSequential(
        SafeModule(
            obs_encoder,
            in_keys=["pixels"],
            out_keys=["encoded_latents"],
        ),
        SafeModule(
            rssm_posterior,
            in_keys=["belief", "encoded_latents"],
            out_keys=[
                "_",
                "_",
                "state",
            ],
        ),
        SafeProbabilisticTensorDictSequential(
            SafeModule(
                actor_module,
                in_keys=["state", "idx", "belief"],
                out_keys=["loc", "scale"],
                spec=CompositeSpec(
                    **{
                        "loc": UnboundedContinuousTensorSpec(
                            proof_environment.action_spec.shape,
                        ),
                        "scale": UnboundedContinuousTensorSpec(
                            proof_environment.action_spec.shape,
                        ),
                    }
                ),
            ),
            SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=[action_key],
                default_interaction_type=InteractionType.MODE,
                distribution_class=distribution_class,
                distribution_kwargs={"tanh_loc": True},
                spec=CompositeSpec(**{action_key: proof_environment.action_spec.to("cpu")}),
            ),
        ),
        SafeModule(
            rssm_prior,
            in_keys=["state", "belief", action_key, "idx"],
            out_keys=[
                "_",
                "_",
                "_",  # we don't need the prior state
                ("next", "belief"),
            ],
        ),
    )
    return actor_realworld


def _dreamer_make_value_model(state_dim, belief_dim, context_model, mlp_num_units, value_key):
    # actor for simulator: interacts with states ~ prior
    critic = Critic(
        state_or_obs_dim=state_dim,
        belief_dim=belief_dim,
        context_dim=context_model.context_dim,
        depth=3,
        num_cells=mlp_num_units,
        activation_class=nn.ELU,
        is_mdp=False,
    )
    critic.set_context_model(context_model)
    value_model = SafeModule(
        critic,
        in_keys=["state", "idx", "belief"],
        out_keys=[value_key],
    )
    return value_model


@dataclass
class DreamerConfig:
    """Dreamer model config struct."""

    batch_length: int = 50

    meta: bool = False

    variable_num: int = 10
    state_dim_per_variable: int = 3
    hidden_dim_per_variable: int = 20
    belief_dim_per_variable: int = 20
    hidden_size: int = 200
    disable_belief: bool = False
    residual: bool = False
    logits_clip: float = 3.0
    max_context_dim: int = 0
    task_num: int = 0

    model_type = "causal"
    using_cross_belief = False
    using_reinforce = True

    actor_dist_type: str = "tanh_normal"

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
