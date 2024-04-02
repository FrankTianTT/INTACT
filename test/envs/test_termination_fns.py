import torch

from intact.envs.termination_fns import termination_fns_dict


def test_termination_fn():
    for fn in termination_fns_dict.values():
        print(f"testing {fn.__name__} ...")
        for batch_size in [(), (10,), (10, 20)]:
            obs = torch.randn(*batch_size, 11)
            act = torch.randn(*batch_size, 3)
            next_obs = torch.randn(*batch_size, 11)
            done = fn(obs, act, next_obs)
            assert done.shape == (*batch_size, 1)
            assert done.dtype == torch.bool
        print("passed")
