from itertools import chain

from tensordict.nn import TensorDictSequential, TensorDictModule

from intact.modules.models.dreamer_world_model.causal_rssm_prior import CausalRSSMPrior
from intact.modules.models.dreamer_world_model.plain_rssm_prior import PlainRSSMPrior


class DreamerWrapper(TensorDictSequential):
    """
    A wrapper class for a sequence of TensorDictModules that represents a Dreamer model.

    This class extends the TensorDictSequential class and adds additional properties and methods
    for handling the components of a Dreamer model.
    """

    def __init__(
        self,
        obs_encoder: TensorDictModule,
        rssm_rollout: TensorDictModule,
        obs_decoder: TensorDictModule,
        reward_model: TensorDictModule,
        continue_model: TensorDictModule = None,
    ):
        """
        Initialize the DreamerWrapper.

        Args:
            obs_encoder (TensorDictModule): The observation encoder module.
            rssm_rollout (TensorDictModule): The RSSM rollout module.
            obs_decoder (TensorDictModule): The observation decoder module.
            reward_model (TensorDictModule): The reward model module.
            continue_model (TensorDictModule, optional): The continue model module. Defaults to None.
        """
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
        """
        Get the observation encoder module.

        Returns:
            TensorDictModule: The observation encoder module.
        """
        return self.module[0]

    @property
    def rssm_rollout(self):
        """
        Get the RSSM rollout module.

        Returns:
            TensorDictModule: The RSSM rollout module.
        """
        return self.module[1]

    @property
    def rssm_posterior(self):
        """
        Get the RSSM posterior module.

        Returns:
            TensorDictModule: The RSSM posterior module.
        """
        return self.rssm_rollout.rssm_posterior.module

    @property
    def obs_decoder(self):
        """
        Get the observation decoder module.

        Returns:
            TensorDictModule: The observation decoder module.
        """
        return self.module[2]

    @property
    def reward_model(self):
        """
        Get the reward model module.

        Returns:
            TensorDictModule: The reward model module.
        """
        return self.module[3]

    @property
    def continue_model(self):
        """
        Get the continue model module.

        Returns:
            TensorDictModule: The continue model module.
        """
        assert self.pred_continue
        return self.module[4]

    @property
    def rssm_prior(self):
        """
        Get the RSSM prior module.

        Returns:
            TensorDictModule: The RSSM prior module.
        """
        return self.rssm_rollout.rssm_prior.module

    @property
    def causal_mask(self):
        """
        Get the causal mask.

        Returns:
            Tensor: The causal mask.
        """
        return self.rssm_prior.causal_mask

    @property
    def context_model(self):
        """
        Get the context model.

        Returns:
            TensorDictModule: The context model.
        """
        return self.rssm_prior.context_model

    def get_parameter(self, target: str):
        """
        Get the parameters of the specified target.

        Args:
            target (str): The target to get the parameters of.

        Returns:
            iterator: An iterator over the parameters of the target.

        Raises:
            NotImplementedError: If the target is not recognized.
        """
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
        """
        Perform a forward pass of the Dreamer model in parallel.

        Args:
            tensordict (TensorDict): The input tensor dictionary.
            sampling_times (int, optional): The number of times to sample. Defaults to 50.

        Returns:
            TensorDict: The output tensor dictionary.
        """
        assert len(tensordict.batch_size) == 2, "batch_size should be 2-d"
        batch_size, batch_len = tensordict.batch_size

        tensordict = self._run_module(self.obs_encoder, tensordict)
        tensordict = tensordict.select(*self.rssm_rollout.in_keys, strict=False)

        repeat_tensordict = tensordict.expand(sampling_times, *tensordict.batch_size).reshape(
            -1, batch_len
        )
        out_tensordict = self._run_module(self.rssm_rollout, repeat_tensordict)
        out_tensordict = out_tensordict.reshape(sampling_times, batch_size, batch_len)

        return out_tensordict
