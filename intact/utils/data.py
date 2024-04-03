import tensordict
import torch


def match_length(batch_td: tensordict.TensorDict, length: int):
    """Match the length of the sequence to the specified length.

    Args:
        batch_td (tensordict.TensorDict): the batch of sequences.
        length (int): the specified length.

    Returns:
        tensordict.TensorDict: the matched batch of sequences.
    """
    assert len(batch_td.shape) == 2, "batch_td must be 2D"

    batch_size, seq_len = batch_td.shape
    # min multiple of length that larger than or equal to seq_len
    new_seq_len = (seq_len + length - 1) // length * length
    # pad the sequence to the new length, add 0 to the end
    matched_td = torch.stack(
        tensors=[tensordict.pad(tensordict=td, pad_size=[0, new_seq_len - seq_len]) for td in batch_td], dim=0
    ).contiguous()
    return matched_td
