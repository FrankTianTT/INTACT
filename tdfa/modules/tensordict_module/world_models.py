from tensordict.nn import TensorDictModule, TensorDictSequential


class CausalDreamerWrapper(TensorDictSequential):
    """World model wrapper.

    This module wraps together a transition model and a reward model.
    The transition model is used to predict an imaginary world state.
    The reward model is used to predict the reward of the imagined transition.

    Args:
        transition_model (TensorDictModule): a transition model that generates a new world states.
        reward_model (TensorDictModule): a reward model, that reads the world state and returns a reward.
        continue_model (TensorDictModule): a continue model, that reads the world state and returns
            a continue probability, optional.

    """

    def __init__(
            self,
            transition_model: TensorDictModule,
            reward_model: TensorDictModule,
            continue_model: TensorDictModule = None,
    ):
        models = [transition_model, reward_model]
        if continue_model is not None:
            models.append(continue_model)
            self.pred_continue = True
        else:
            self.pred_continue = False

        super().__init__(*models)

    def get_transition_model_operator(self) -> TensorDictModule:
        """Returns a transition operator that maps either an observation to a world state or a world state to the next world state."""
        return self.module[0]

    def get_reward_operator(self) -> TensorDictModule:
        """Returns a reward operator that maps a world state to a reward."""
        return self.module[1]

    def get_continue_operator(self) -> TensorDictModule:
        """Returns a continue operator that maps a world state to a continue probability."""
        if self.pred_continue:
            return self.module[2]
        else:
            raise NotImplementedError("continue model is not defined")

    def get_parameter(self, target: str):
        if target == "module":
            for name, param in self.named_parameters(recurse=True):
                if "context_hat" not in name and "mask_logits" not in name:
                    yield param
        elif target == "context":
            for name, param in self.named_parameters(recurse=True):
                if "context_hat" in name:
                    yield param
        elif target == "mask_logits":
            for name, param in self.named_parameters(recurse=True):
                if "mask_logits" in name:
                    yield param
        else :
            raise NotImplementedError
