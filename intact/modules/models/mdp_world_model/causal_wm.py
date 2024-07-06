import torch
from torch.nn import ModuleDict

from intact.modules.models.causal_mask import CausalMask
from intact.modules.models.mdp_world_model.plain_wm import PlainMDPWorldModel
from intact.modules.utils import build_mlp


class CausalWorldModel(PlainMDPWorldModel):
    def __init__(
        self,
        obs_dim,
        action_dim,
        meta=False,
        mask_type="direct",
        alpha=10.0,
        max_context_dim=10,
        task_num=100,
        residual=True,
        hidden_dims=None,
        log_var_bounds=(-10.0, 0.5),
        logits_clip=3.0,
        observed_logits_init_bias=0.5,
        context_logits_init_bias=0.5,
        logits_init_scale=0.0,
        sigmoid_threshold=0.1,
    ):
        """Initializes the CausalWorldModel class.

        Args:
            obs_dim (int): Number of observation dimensions.
            action_dim (int): Number of action dimensions.
            meta (bool, optional): Whether to use envs-RL. Defaults to False.
            alpha (float, optional): Alpha parameter for the CausalMask class. Defaults to 10.0.
            max_context_dim (int, optional): Number of context dimensions, used for envs-RL. Defaults to 10.
            task_num (int, optional): Number of tasks, used for envs-RL. Defaults to 100.
            residual (bool, optional): Whether to use residual connection for transition model. Defaults to True.
            hidden_dims (list, optional): Hidden dimensions for transition model. Defaults to None.
            log_var_bounds (tuple, optional): Bounds for log_var of gaussian nll loss. Defaults to (-10.0, 0.5).
            logits_clip (float, optional): Clip value for mask logits. Defaults to 3.0.
            observed_logits_init_bias (float, optional): Bias for mask logits for observed variables initialization. Defaults to 0.5.
            context_logits_init_bias (float, optional): Bias for mask logits for context variables initialization. Defaults to 0.5.
            logits_init_scale (float, optional): Scale for mask logits initialization. Defaults to 0.0.
        """
        self.mask_type = mask_type
        self.logits_clip = logits_clip
        self.observed_logits_init_bias = observed_logits_init_bias
        self.context_logits_init_bias = context_logits_init_bias
        self.logits_init_scale = logits_init_scale
        self.sigmoid_threshold = sigmoid_threshold

        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            meta=meta,
            max_context_dim=max_context_dim,
            task_num=task_num,
            residual=residual,
            hidden_dims=hidden_dims,
            log_var_bounds=log_var_bounds,
        )

        self.causal_mask = CausalMask(
            observed_input_dim=self.obs_dim + self.action_dim,
            mask_output_dim=self.output_dim,
            mask_type=self.mask_type,
            meta=self.meta,
            context_input_dim=self.max_context_dim,
            logits_clip=self.logits_clip,
            alpha=alpha,
            sigmoid_threshold=self.sigmoid_threshold,
        )

    def build_nets(self):
        """Builds the neural networks for the model.

        Returns:
            ModuleDict: A dictionary of the neural networks.
        """
        para_mlp = build_mlp(
            input_dim=self.all_input_dim,
            output_dim=2,
            hidden_dims=self.hidden_dims,
            extra_dims=[self.output_dim],
            activate_name="ReLU",
        )
        return ModuleDict(dict(para_mlp=para_mlp))

    @property
    def params_dict(self):
        """Gets the parameters of the model.

        Returns:
            dict: A dictionary of the parameters.
        """
        return dict(
            nets=self.nets.parameters(),
            context=self.context_model.parameters(),
            observed_logits=self.causal_mask.get_parameter("observed_logits"),
            context_logits=self.causal_mask.get_parameter("context_logits"),
        )

    def forward(self, observation, action, idx=None, deterministic_mask=False):
        """Performs a forward pass through the model.

        Args:
            observation (Tensor): The observations.
            action (Tensor): The actions.
            idx (int, optional): The index. Defaults to None.
            deterministic_mask (bool, optional): Whether to use a deterministic mask. Defaults to False.

        Returns:
            tuple: The outputs of the forward pass.
        """
        inputs = torch.cat([observation, action, self.context_model(idx)], dim=-1)
        batch_shape, dim = inputs.shape[:-1], inputs.shape[-1]

        masked_inputs, mask = self.causal_mask(inputs.reshape(-1, dim), deterministic=deterministic_mask)

        mean, log_var = self.nets["para_mlp"](masked_inputs).permute(2, 1, 0)

        mask = mask.reshape(
            *batch_shape,
            self.causal_mask.mask_output_dim,
            self.causal_mask.mask_input_dim,
        )
        return *self.get_outputs(mean, log_var, observation, batch_shape), mask
