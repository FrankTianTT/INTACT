from itertools import chain

import torch
from tensordict.nn import TensorDictSequential, TensorDictModule
from torchrl.modules import SafeModule
from torchrl.modules.models.model_based import ObsDecoder, ObsEncoder, RSSMPosterior, RSSMRollout
from torchrl.modules.models.models import MLP

from causal_meta.modules.models.dreamer_world_model.causal_rssm_prior import CausalRSSMPrior
from causal_meta.modules.models.dreamer_world_model.plain_rssm_prior import PlainRSSMPrior


class DreamerWrapper(TensorDictSequential):
    def __init__(
        self,
        obs_encoder: TensorDictModule,
        rssm_rollout: TensorDictModule,
        obs_decoder: TensorDictModule,
        reward_model: TensorDictModule,
        continue_model: TensorDictModule = None,
    ):

        self.variable_num = rssm_rollout.rssm_prior.variable_num
        self.state_dim_per_variable = rssm_rollout.rssm_prior.state_dim_per_variable
        self.hidden_dim_per_variable = rssm_rollout.rssm_prior.belief_dim_per_variable
        self.action_dim = rssm_rollout.rssm_prior.action_dim

        models = [obs_encoder, rssm_rollout, obs_decoder, reward_model]
        if continue_model is not None:
            models.append(continue_model)
            self.pred_continue = True
        else:
            self.pred_continue = False

        super().__init__(*models)

        if isinstance(self.rssm_prior, PlainRSSMPrior):
            if isinstance(self.rssm_prior, CausalRSSMPrior):
                self.model_type = "causal"
            else:
                self.model_type = "plain"
        else:
            raise NotImplementedError

    @property
    def obs_encoder(self):
        return self.module[0]

    @property
    def rssm_rollout(self):
        return self.module[1]

    @property
    def rssm_posterior(self):
        return self.rssm_rollout.rssm_posterior.module

    @property
    def obs_decoder(self):
        return self.module[2]

    @property
    def reward_model(self):
        return self.module[3]

    @property
    def continue_model(self):
        assert self.pred_continue
        return self.module[4]

    @property
    def rssm_prior(self):
        return self.rssm_rollout.rssm_prior.module

    @property
    def causal_mask(self):
        return self.rssm_prior.causal_mask

    @property
    def context_model(self):
        return self.rssm_prior.context_model

    def get_parameter(self, target: str):
        if target == "nets":
            return chain(
                self.obs_encoder.parameters(),
                self.rssm_prior.get_parameter("nets"),
                self.rssm_posterior.parameters(),
                self.obs_decoder.parameters(),
                self.reward_model.parameters(),
                self.continue_model.parameters() if self.pred_continue else [],
            )
        elif target == "context":
            return self.rssm_prior.get_parameter("context")
        elif target == "observed_logits":
            return self.rssm_prior.get_parameter("observed_logits")
        elif target == "context_logits":
            return self.rssm_prior.get_parameter("context_logits")
        else:
            raise NotImplementedError

    def parallel_forward(self, tensordict, sampling_times=50):
        assert len(tensordict.batch_size) == 2, "batch_size should be 2-d"
        batch_size, batch_len = tensordict.batch_size

        tensordict = self._run_module(self.obs_encoder, tensordict)
        tensordict = tensordict.select(*self.rssm_rollout.in_keys, strict=False)

        repeat_tensordict = tensordict.expand(sampling_times, *tensordict.batch_size).reshape(-1, batch_len)
        out_tensordict = self._run_module(self.rssm_rollout, repeat_tensordict)
        out_tensordict = out_tensordict.reshape(sampling_times, batch_size, batch_len)

        return out_tensordict
