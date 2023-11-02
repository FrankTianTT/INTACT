import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torchrl.objectives.common import LossModule
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase

from tdfa.models.util import build_parallel_layers


class CausalWorldModel(TensorDictModuleBase):
    def __init__(
            self,
            obs_dim,
            action_dim,
            max_context_dim=0,
            task_num=0,
            residual=True,
            logits_clip=4.0,
            stochastic=True,
            logvar_bounds=(-10.0, 0.5)
    ):
        """World-model class for environment learning with causal discovery.

        :param obs_dim: number of observation dimensions
        :param action_dim: number of action dimensions
        :param max_context_dim: number of context dimensions, used for meta-RL, set to 0 if normal RL
        :param task_num: number of tasks, used for meta-RL, set to 0 if normal RL
        :param residual: whether to use residual connection for transition model
        :param logits_clip: clip value for mask logits
        :param stochastic: whether to use stochastic transition model (using gaussian nll loss rather than mse loss)
        :param logvar_bounds: bounds for logvar of gaussian nll loss
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.context_dim = max_context_dim
        self.task_num = task_num
        self.residual = residual
        self.logits_clip = logits_clip
        self.stochastic = stochastic
        self.logvar_bounds = logvar_bounds

        self.in_keys = ["observation", "action", "idx"]
        self.out_keys = ["observation", "reward", "terminated"]

        self.module = build_parallel_layers(
            self.all_input_dim,
            2 if self.stochastic else 1,
            [64, 64],
            extra_dims=[self.output_dim],
            activate_name="ReLu",
        )
        self._mask_logits = nn.Parameter(torch.randn(self.output_dim, self.all_input_dim))
        self.context_hat = torch.nn.Parameter(torch.randn(task_num, max_context_dim))

    @property
    def is_meta(self):
        return self.context_dim > 0 and self.task_num > 0

    @property
    def mask_logits(self):
        return torch.clamp(self._mask_logits, -self.logits_clip, self.logits_clip)

    def get_parameter(self, target: str):
        if target == "module":
            return self.module.parameters()
        elif target == "mask_logits":
            return [self._mask_logits]
        elif target == "context_hat":
            return [self.context_hat]
        else:
            raise NotImplementedError

    @property
    def all_input_dim(self):
        return self.obs_dim + self.action_dim + self.context_dim

    @property
    def valid_input_dim(self):
        return self.obs_dim + self.action_dim

    @property
    def output_dim(self):
        return self.obs_dim + 2

    def module_forward(self, observation, action, idx, deterministic_mask=False):
        batch_size, _ = observation.shape
        if self.is_meta:
            inputs = torch.cat([observation, action, self.context_hat[idx]], dim=-1)
        else:
            inputs = torch.cat([observation, action], dim=-1)
        repeated_inputs = inputs.unsqueeze(0).expand(self.output_dim, -1, -1)  # o, b, i

        if deterministic_mask:
            mask = torch.gt(self.mask_logits, 0).float().expand(batch_size, -1, -1)
        else:
            mask = Bernoulli(logits=self.mask_logits).sample(torch.Size([batch_size]))  # b, o, i
        masked_inputs = torch.einsum("boi,obi->obi", mask, repeated_inputs)

        outputs = self.module(masked_inputs).permute(2, 1, 0)

        do = outputs[0]  # deterministic outputs
        next_observation, reward, terminated = do[:, :-2], do[:, -2:-1], do[:, -1:]

        if self.residual:
            next_observation = observation + next_observation

        if self.stochastic:
            logvar = outputs[1, :, :-2]
            min_logvar, max_logvar = self.logvar_bounds
            logvar = max_logvar - F.softplus(max_logvar - logvar)
            logvar = min_logvar + F.softplus(logvar - min_logvar)
            next_observation = (next_observation, logvar)

        return next_observation, reward, terminated, mask

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Using for rollout, rather than training.

        :param tensordict: data with "observation", "action" and "idx" (optional).
        :return:
            tensordict: added ("next", "observation"), ("next", "reward") and ("next", "terminated").
        """
        observation = tensordict.get("observation")
        action = tensordict.get("action")
        idx = tensordict.get("idx") if self.is_meta else None
        next_observation, reward, terminated, mask = self.module_forward(observation, action, idx,
                                                                         deterministic_mask=True)
        if self.stochastic:  # sample
            mean, logvar = next_observation
            std = torch.exp(0.5 * logvar)
            tensordict.set("observation", mean + std * torch.randn_like(std))
        else:
            tensordict.set("observation", next_observation)
        tensordict.set("reward", reward)
        tensordict.set("terminated", terminated)
        return tensordict


class CausalWorldModelLoss(LossModule):
    def __init__(
            self,
            world_model: CausalWorldModel,
            lambda_transition: float = 1.0,
            lambda_reward: float = 1.0,
            lambda_terminated: float = 1.0,
            sparse_weight: float = 0.05,
            context_weight: float = 0.1,
            sampling_times: int = 50,
    ):
        super().__init__()
        self.world_model = world_model
        self.valid_input_dim = world_model.valid_input_dim
        self.all_input_dim = world_model.all_input_dim
        self.output_dim = world_model.output_dim

        self.lambda_transition = lambda_transition
        self.lambda_reward = lambda_reward
        self.lambda_terminated = lambda_terminated
        self.sparse_weight = sparse_weight
        self.context_weight = context_weight
        self.sampling_times = sampling_times

    def forward(self, tensordict: TensorDict, intermediate=False):
        observation = tensordict.get("observation")
        action = tensordict.get("action")
        idx = tensordict.get("idx") if self.world_model.is_meta else None

        pred_observation, pred_reward, pred_terminated, mask = self.world_model.module_forward(observation, action, idx)
        if self.world_model.stochastic:
            mean, logvar = pred_observation
            transition_loss = F.gaussian_nll_loss(
                mean,
                tensordict.get(("next", "observation")),
                torch.exp(logvar),
                reduction="none"
            ) * self.lambda_transition
        else:
            transition_loss = F.mse_loss(
                pred_observation,
                tensordict.get(("next", "observation")),
                reduction="none"
            ) * self.lambda_transition
        reward_loss = F.mse_loss(
            pred_reward,
            tensordict.get(("next", "reward")),
            reduction="none"
        ) * self.lambda_reward
        terminated_loss = F.binary_cross_entropy_with_logits(
            pred_terminated,
            tensordict.get(("next", "terminated")).float(),
            reduction="none"
        ) * self.lambda_terminated
        all_loss = torch.cat([transition_loss, reward_loss, terminated_loss], dim=-1)

        if intermediate:
            return all_loss, mask
        else:
            return all_loss

    def get_mask_grad(self, sampling_loss, sampling_mask, eps=1e-6):
        num_pos = sampling_mask.sum(dim=0)
        num_neg = sampling_mask.shape[0] - num_pos
        is_valid = ((num_pos > 0) * (num_neg > 0)).float()
        pos_grads = torch.einsum("sbo,sboi->sboi", sampling_loss, sampling_mask).sum(dim=0) / (num_pos + eps)
        neg_grads = torch.einsum("sbo,sboi->sboi", sampling_loss, 1 - sampling_mask).sum(dim=0) / (num_neg + eps)

        logits = self.world_model.mask_logits
        g = logits.sigmoid() * (1 - logits.sigmoid())
        reg = torch.ones(pos_grads.shape[1:]) * self.sparse_weight
        reg[:, self.valid_input_dim:] += self.context_weight
        grad = is_valid * (pos_grads - neg_grads + reg) * g
        return grad.mean(0)

    def reinforce(self, tensordict: TensorDict):
        tensordict = tensordict.clone(recurse=False)
        repeated_tensordict = tensordict.expand(self.sampling_times, *tensordict.batch_size).reshape(-1)
        loss, mask = self.forward(repeated_tensordict, intermediate=True)

        sampling_loss = loss.reshape(self.sampling_times, -1, self.output_dim)
        sampling_mask = mask.reshape(self.sampling_times, -1, self.output_dim, self.all_input_dim)
        mask_grad = self.get_mask_grad(sampling_loss, sampling_mask)
        return mask_grad


if __name__ == '__main__':
    causal_world_model = CausalWorldModel(4, 1, 0, 3)

    obs = torch.randn(5, 4)
    action = torch.randn(5, 1)
    idx = torch.randint(0, 3, (5,))

    tensordict = TensorDict(
        {
            "observation": obs,
            "action": action,
            "idx": idx,
            "next": {
                "observation": torch.randn(5, 4),
                "reward": torch.randn(5, 1),
                "terminated": torch.randint(0, 2, (5, 1)),
            }}, batch_size=5)

    world_model_loss = CausalWorldModelLoss(causal_world_model)
    world_model_loss.reinforce(tensordict)
