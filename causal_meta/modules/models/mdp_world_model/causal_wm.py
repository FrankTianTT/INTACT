import torch
from torch.nn import ModuleDict

from causal_meta.modules.utils import build_mlp
from causal_meta.modules.models.mdp_world_model.plain_wm import PlainMDPWorldModel
from causal_meta.modules.models.causal_mask import CausalMask


class CausalWorldModel(PlainMDPWorldModel):
    def __init__(
            self,
            obs_dim,
            action_dim,
            meta=False,
            reinforce=True,
            alpha=10.,
            max_context_dim=10,
            task_num=100,
            residual=True,
            hidden_dims=None,
            log_var_bounds=(-10.0, 0.5),
            logits_clip=3.0,
            observed_logits_init_bias=0.5,
            context_logits_init_bias=0.5,
            logits_init_scale=0.0,
    ):
        """World-model class for environment learning with causal discovery.

        :param obs_dim: number of observation dimensions
        :param action_dim: number of action dimensions
        :param meta: whether to use envs-RL
        :param max_context_dim: number of context dimensions, used for envs-RL, set to 0 if normal RL
        :param task_num: number of tasks, used for envs-RL, set to 0 if normal RL
        :param residual: whether to use residual connection for transition model
        :param hidden_dims: hidden dimensions for transition model
        :param log_var_bounds: bounds for log_var of gaussian nll loss
        :param logits_clip: clip value for mask logits, default to 3.0 (sigmoid(3.0) = 0.95).
        :param observed_logits_init_bias: bias for mask logits for observed variables initialization,
            default to 0.5 (sigmoid(0.5) = 0.62).
        :param context_logits_init_bias: bias for mask logits for context variables initialization,
            default to 0.5 (sigmoid(0.5) = 0.62).
        :param logits_init_scale: scale for mask logits initialization, default to 0.05
        :param log_var_bounds: bounds for log_var of gaussian nll loss
        """
        self.reinforce = reinforce
        self.logits_clip = logits_clip
        self.observed_logits_init_bias = observed_logits_init_bias
        self.context_logits_init_bias = context_logits_init_bias
        self.logits_init_scale = logits_init_scale

        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            meta=meta,
            max_context_dim=max_context_dim,
            task_num=task_num,
            residual=residual,
            hidden_dims=hidden_dims,
            log_var_bounds=log_var_bounds
        )

        self.causal_mask = CausalMask(
            observed_input_dim=self.obs_dim + self.action_dim,
            mask_output_dim=self.output_dim,
            reinforce=self.reinforce,
            meta=self.meta,
            context_input_dim=self.max_context_dim,
            logits_clip=self.logits_clip,
            alpha=alpha
            # observed_logits_init_bias=observed_logits_init_bias,
            # context_logits_init_bias=context_logits_init_bias,
            # logits_init_scale=logits_init_scale,
        )

    def build_nets(self):
        para_mlp = build_mlp(
            input_dim=self.all_input_dim,
            output_dim=2,
            hidden_dims=self.hidden_dims,
            extra_dims=[self.output_dim],
            activate_name="SiLU",
        )
        return ModuleDict(dict(para_mlp=para_mlp))

    # def build_nets(self):
    #     para_mlp1 = build_mlp(
    #         input_dim=self.obs_dim + self.action_dim ,
    #         output_dim=self.hidden_dims[-2],
    #         hidden_dims=self.hidden_dims[:-2],
    #         extra_dims=[self.output_dim],
    #         activate_name="ReLU",
    #     )
    #     para_mlp2 = build_mlp(
    #         input_dim=self.hidden_dims[-2] + self.max_context_dim,
    #         output_dim=2,
    #         hidden_dims=[self.hidden_dims[-1]],
    #         extra_dims=[self.output_dim],
    #         activate_name="SiLU",
    #     )
    #     return ModuleDict(dict(para_mlp1=para_mlp1, para_mlp2=para_mlp2))

    @property
    def params_dict(self):
        return dict(
            nets=self.nets.parameters(),
            context=self.context_model.parameters(),
            observed_logits=self.causal_mask.get_parameter("observed_logits"),
            context_logits=self.causal_mask.get_parameter("context_logits"),
        )

    def forward(self, observation, action, idx=None, deterministic_mask=False):
        inputs = torch.cat([observation, action, self.context_model(idx)], dim=-1)
        batch_shape, dim = inputs.shape[:-1], inputs.shape[-1]

        masked_inputs, mask = self.causal_mask(inputs.reshape(-1, dim), deterministic=deterministic_mask)
        # first_inputs = masked_inputs[:, :, :self.obs_dim + self.action_dim]
        # hidden = self.nets["para_mlp1"](first_inputs)
        #
        # second_inputs = torch.cat([hidden, masked_inputs[:, :, self.obs_dim + self.action_dim:]], dim=-1)
        # mean, log_var = self.nets["para_mlp2"](second_inputs).permute(2, 1, 0)

        mean, log_var = self.nets["para_mlp"](masked_inputs).permute(2, 1, 0)

        mask = mask.reshape(*batch_shape, self.causal_mask.mask_output_dim, self.causal_mask.mask_input_dim)
        return *self.get_outputs(mean, log_var, observation, batch_shape), mask


def test_causal_world_model_without_meta():
    obs_dim = 4
    action_dim = 1
    batch_size = 32
    env_num = 5

    world_model = CausalWorldModel(obs_dim=obs_dim, action_dim=action_dim, meta=False)

    for batch_shape in [(), (batch_size,), (env_num, batch_size)]:
        observation = torch.randn(*batch_shape, obs_dim)
        action = torch.randn(*batch_shape, action_dim)

        next_obs_mean, next_obs_log_var, reward, terminated, mask = world_model(observation, action)

        assert next_obs_mean.shape == next_obs_log_var.shape == (*batch_shape, obs_dim)
        assert reward.shape == terminated.shape == (*batch_shape, 1)
        assert mask.shape == (*batch_shape, obs_dim + 2, obs_dim + action_dim)


def test_causal_world_model_with_meta():
    obs_dim = 4
    action_dim = 1
    max_context_dim = 10
    task_num = 100
    batch_size = 32
    env_num = 5

    world_model = CausalWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        meta=True,
        max_context_dim=max_context_dim,
        task_num=task_num
    )

    for batch_shape in [(), (batch_size,), (env_num, batch_size)]:
        observation = torch.randn(*batch_shape, obs_dim)
        action = torch.randn(*batch_shape, action_dim)
        idx = torch.randint(0, task_num, (*batch_shape, 1))

        next_obs_mean, next_obs_log_var, reward_mean, reward_log_var, terminated, mask \
            = world_model(observation, action, idx)

        assert next_obs_mean.shape == next_obs_log_var.shape == (*batch_shape, obs_dim)
        assert reward_mean.shape == reward_log_var.shape == terminated.shape == (*batch_shape, 1)
        assert mask.shape == (*batch_shape, obs_dim + 2, obs_dim + action_dim + max_context_dim)


def test_reset():
    obs_dim = 4
    action_dim = 1
    max_context_dim = 10
    task_num = 100

    world_model = CausalWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        meta=True,
        max_context_dim=max_context_dim,
        task_num=task_num
    )
    world_model.reset()


if __name__ == '__main__':
    test_causal_world_model_with_meta()
