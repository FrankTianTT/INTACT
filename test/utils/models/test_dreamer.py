import torch
from torchrl.envs import GymEnv, TransformedEnv, Compose, ToTensorImage, DoubleToFloat, TensorDictPrimer
from torchrl.data import UnboundedContinuousTensorSpec

from causal_meta.utils.models.mdp import DreamerConfig
from causal_meta.utils.models.dreamer import make_dreamer


# def test_make_causal_dreamer():
#     cfg = DreamerConfig()
#
#     env = TransformedEnv(GymEnv("MyCartPole-v0", from_pixels=True), Compose(ToTensorImage(), DoubleToFloat()))
#
#     default_dict = {
#         "state": UnboundedContinuousTensorSpec(
#             shape=torch.Size((*env.batch_size, cfg.variable_num * cfg.state_dim_per_variable))
#         ),
#         "belief": UnboundedContinuousTensorSpec(
#             shape=torch.Size((*env.batch_size, cfg.variable_num * cfg.hidden_dim_per_variable))
#         ),
#     }
#     env.append_transform(TensorDictPrimer(random=False, default_value=0, **default_dict))
#
#     world_model, model_based_env, actor_simulator, value_model, actor_realworld = make_dreamer(cfg, env)

# world_model.get_parameter("transition_model.0.weight")
