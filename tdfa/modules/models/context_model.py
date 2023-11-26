import torch
from torch import nn


class ContextModel(nn.Module):
    def __init__(
            self,
            meta=False,
            max_context_dim=0,
            task_num=0,
    ):
        super().__init__()
        self.meta = meta
        self.max_context_dim = max_context_dim
        self.task_num = task_num

        self.context_hat = torch.nn.Parameter(torch.randn(task_num, max_context_dim))

    def forward(self, idx=None):
        if idx is None:
            assert not self.meta, "idx should not be None when meta is True"
            return torch.empty(0)

        assert idx.shape[-1] == 1, "last dim of idx should be 1, got {}".format(idx.shape)
        return self.context_hat[idx[..., 0]]


def test_context_model():
    max_context_dim = 10
    task_num = 100
    batch_size = 32
    x_size = 4

    context_model = ContextModel(meta=False)
    x = torch.randn(batch_size, x_size)
    x = torch.cat([x, context_model(None)], dim=-1)
    assert x.shape == (batch_size, x_size)

    context_model = ContextModel(meta=True, max_context_dim=max_context_dim, task_num=task_num)
    idx = torch.randint(0, task_num, (batch_size, 1))
    x = torch.randn(batch_size, x_size)
    x = torch.cat([x, context_model(idx)], dim=-1)
    assert x.shape == (batch_size, x_size + max_context_dim)


if __name__ == '__main__':
    test_context_model()
