import torch
from torchrl.modules import SafeModule
from torchrl.modules.models.model_based import ObsDecoder, ObsEncoder, RSSMPosterior, RSSMRollout
from torchrl.modules.models.models import MLP

from intact.modules.models.dreamer_world_model.causal_rssm_prior import CausalRSSMPrior
from intact.modules.tensordict_module.dreamer_wrapper import DreamerWrapper


def build_example_causal_dreamer_wrapper(meta=False):
    from torch import nn

    action_dim = 1
    variable_num = 10
    state_dim_per_variable = 3
    hidden_dim_per_variable = 20
    max_context_dim = 10 if meta else 0
    task_num = 100 if meta else 0
    mlp_num_units = 200

    obs_encoder = ObsEncoder()
    obs_decoder = ObsDecoder()

    rssm_prior = CausalRSSMPrior(
        action_dim=action_dim,
        variable_num=variable_num,
        state_dim_per_variable=state_dim_per_variable,
        belief_dim_per_variable=hidden_dim_per_variable,
        max_context_dim=max_context_dim,
        task_num=task_num,
    )
    rssm_posterior = RSSMPosterior(
        hidden_dim=hidden_dim_per_variable * variable_num,
        state_dim=state_dim_per_variable * variable_num,
    )
    reward_module = MLP(out_features=1, depth=2, num_cells=mlp_num_units, activation_class=nn.ELU)

    continue_module = MLP(out_features=1, depth=2, num_cells=mlp_num_units, activation_class=nn.ELU)

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


def get_example_data(meta=False):
    import torch
    from tensordict import TensorDict

    action_dim = 1
    variable_num = 10
    state_dim_per_variable = 3
    hidden_dim_per_variable = 20
    batch_size = 8
    batch_len = 10
    task_num = 100 if meta else 0

    input_td = TensorDict(
        {
            "state": torch.randn(batch_size, batch_len, variable_num * state_dim_per_variable),
            "belief": torch.randn(batch_size, batch_len, variable_num * hidden_dim_per_variable),
            "action": torch.randn(batch_size, batch_len, action_dim),
            "next": {
                "pixels": torch.randn(batch_size, batch_len, 3, 64, 64),
                "state": torch.randn(batch_size, batch_len, variable_num * state_dim_per_variable),
                "belief": torch.randn(
                    batch_size, batch_len, variable_num * hidden_dim_per_variable
                ),
            },
        },
        batch_size=(batch_size, batch_len),
    )

    if meta:
        idx = torch.randint(0, task_num, (batch_size, 1))
        idx = idx.reshape(-1, 1, 1).expand(batch_size, batch_len, 1)
        input_td.set("idx", idx)

    return input_td


def test_optimize():
    from torch.nn.functional import binary_cross_entropy_with_logits

    world_model = build_example_causal_dreamer_wrapper()
    input_td = get_example_data()

    output_td = world_model(input_td)

    pred_continue = output_td.get(("next", "pred_continue"))
    terminated = torch.randint(0, 2, pred_continue.shape).bool()

    continue_loss = binary_cross_entropy_with_logits(
        pred_continue, 1 - terminated.float(), reduction="mean"
    )

    continue_loss.backward()

    for param in world_model.get_parameter("nets"):
        if param.grad is not None:
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    for name, p in world_model.named_parameters():
        print(name)
