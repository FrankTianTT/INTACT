# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from tensordict.nn import InteractionType
from torch import distributions as d, nn

from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.common import EnvBase
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import TensorDictPrimer, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    NoisyLinear,
    NormalParamWrapper,
    SafeModule,
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
    SafeSequential,
)
from torchrl.modules.distributions import (
    Delta,
    OneHotCategorical,
    TanhDelta,
    TanhNormal,
)
from torchrl.modules.distributions.continuous import SafeTanhTransform
from torchrl.modules.models.exploration import LazygSDEModule
from torchrl.modules.models.model_based import (
    DreamerActor,
    ObsDecoder,
    ObsEncoder,
    RSSMPosterior,
    RSSMPrior,
    RSSMRollout,
)
from torchrl.modules.models.models import (
    DdpgCnnActor,
    DdpgCnnQNet,
    DuelingCnnDQNet,
    DuelingMlpDQNet,
    MLP,
)
from torchrl.modules.tensordict_module import (
    Actor,
    DistributionalQValueActor,
    QValueActor,
)
from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
from torchrl.modules.tensordict_module.world_models import WorldModelWrapper
from torchrl.trainers.helpers import transformed_env_constructor
from torchrl.trainers.helpers.models import _dreamer_make_world_model, _dreamer_make_mbenv, _dreamer_make_value_model, \
    _dreamer_make_actors

from tdfa.dreamer.causal_rssm_prior import CausalRSSMPrior


def make_causal_dreamer(
        cfg: "DictConfig",  # noqa: F821
        proof_environment: EnvBase = None,
        device: DEVICE_TYPING = "cpu",
        action_key: str = "action",
        value_key: str = "state_value",
        use_decoder_in_env: bool = False,
        obs_norm_state_dict=None,
) -> nn.ModuleList:
    # Modules
    obs_encoder = ObsEncoder()
    obs_decoder = ObsDecoder()

    rssm_prior = CausalRSSMPrior(
        action_dim=proof_environment.action_space.shape[0],
        variable_num=cfg.variable_num,
        state_dim_per_variable=cfg.state_dim_per_variable,
        hidden_dim_per_variable=cfg.hidden_dim_per_variable,
        rnn_input_dim_per_variable=cfg.rnn_input_dim_per_variable,
        max_context_dim=cfg.max_context_dim,
        task_num=cfg.task_num,
        residual=cfg.residual,
        logits_clip=cfg.logits_clip,
        scale_lb=cfg.scale_lb,
    )
    rssm_posterior = RSSMPosterior(
        hidden_dim=cfg.hidden_dim_per_variable * cfg.variable_num,
        state_dim=cfg.state_dim_per_variable * cfg.variable_num,
    )
    reward_module = MLP(
        out_features=1, depth=2, num_cells=cfg.mlp_num_units, activation_class=nn.ELU
    )

    world_model = _dreamer_make_world_model(
        obs_encoder, obs_decoder, rssm_prior, rssm_posterior, reward_module
    ).to(device)
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        tensordict = proof_environment.fake_tensordict().unsqueeze(-1)
        tensordict = tensordict.to_tensordict().to(device)
        tensordict = tensordict.reshape(1, 1)  # batch-size * time-step
        world_model(tensordict)

    model_based_env = _dreamer_make_mbenv(
        reward_module,
        rssm_prior,
        obs_decoder,
        proof_environment,
        use_decoder_in_env,
        cfg.state_dim,
        cfg.rssm_hidden_dim,
    )
    model_based_env = model_based_env.to(device)

    actor_simulator, actor_realworld = _dreamer_make_actors(
        obs_encoder,
        rssm_prior,
        rssm_posterior,
        cfg.mlp_num_units,
        action_key,
        proof_environment,
    )
    actor_simulator = actor_simulator.to(device)

    value_model = _dreamer_make_value_model(cfg.mlp_num_units, value_key)
    value_model = value_model.to(device)
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        tensordict = model_based_env.fake_tensordict().unsqueeze(-1)
        tensordict = tensordict.to(device)
        tensordict = actor_simulator(tensordict)
        value_model(tensordict)

    actor_realworld = actor_realworld.to(device)

    del tensordict
    return world_model, model_based_env, actor_simulator, value_model, actor_realworld


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig, OmegaConf


    @hydra.main(version_base="1.1", config_path=".", config_name="config")
    def _main(cfg: DictConfig) -> None:
        world_model, model_based_env, actor_simulator, value_model, actor_realworld = make_dreamer(cfg)
        print(world_model)


    _main()
