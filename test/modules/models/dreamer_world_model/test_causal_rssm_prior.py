import torch
from causal_meta.modules.models.dreamer_world_model.causal_rssm_prior import CausalRSSMPrior


def test_causal_rssm_prior():
    action_dim = 1
    variable_num = 10
    state_dim_per_variable = 3
    hidden_dim_per_variable = 20
    max_context_dim = 10
    task_num = 50
    env_num = 10
    batch_size = 32

    prior = CausalRSSMPrior(
        action_dim=action_dim,
        variable_num=variable_num,
        state_dim_per_variable=state_dim_per_variable,
        belief_dim_per_variable=hidden_dim_per_variable,
        max_context_dim=max_context_dim,
        task_num=task_num,
        meta=True,
    )

    for batch_shape in [(), (batch_size,), (env_num, batch_size)]:
        state = torch.randn(*batch_shape, variable_num * state_dim_per_variable)
        belief = torch.randn(*batch_shape, variable_num * hidden_dim_per_variable)
        action = torch.randn(*batch_shape, action_dim)
        idx = torch.randint(0, task_num, (*batch_shape, 1))

        prior_mean, prior_std, next_state, next_belief, mask = prior(state, belief, action, idx)

        assert prior_mean.shape == prior_std.shape == next_state.shape == state.shape
        assert next_belief.shape == belief.shape
