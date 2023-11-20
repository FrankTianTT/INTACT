import torch
from torch import nn


class ContextModel(nn.Module):
    def __init__(
            self,
            max_context_dim=0,
            task_num=0,
    ):
        super().__init__()
        self.max_context_dim = max_context_dim
        self.task_num = task_num

        self.context_hat = torch.nn.Parameter(torch.randn(task_num, max_context_dim))

    @property
    def is_meta(self):
        return self.max_context_dim > 0 and self.task_num > 0

    def forward(self, idx):
        assert len(idx.shape) == 2 and idx.shape[1] == 1
        batch_size = idx.shape[0]

        if self.is_meta:
            return self.context_hat[idx.reshape(-1)]
        else:
            return torch.empty(batch_size, 0).to(self.context_hat.device)


if __name__ == '__main__':
    max_context_dim = 10
    task_num = 10
    batch_size = 5

    context_model = ContextModel(
        max_context_dim=max_context_dim,
        task_num=task_num,
    )

    idx = torch.randint(0, task_num, (batch_size, 1))

    context = context_model(idx)

    print(context.shape)

    print(context_model.parameters())
