import torch
import tensordict


def grad_norm(optimizer: torch.optim.Optimizer):
    sum_of_sq = 0.0
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            sum_of_sq += p.grad.pow(2).sum()
    return sum_of_sq.sqrt().detach().item()


def retrieve_stats_from_state_dict(obs_norm_state_dict):
    return {
        "loc": obs_norm_state_dict["loc"],
        "scale": obs_norm_state_dict["scale"],
    }


def match_length(batch_td: tensordict.TensorDict, length):
    assert len(batch_td.shape) == 2, "batch_td must be 2D"

    batch_size, seq_len = batch_td.shape
    # min multiple of length that larger than or equal to seq_len
    new_seq_len = (seq_len + length - 1) // length * length

    # pad with zeros
    matched_td = torch.stack(
        [tensordict.pad(td, [0, new_seq_len - seq_len]) for td in batch_td], 0
    ).contiguous()
    return matched_td


