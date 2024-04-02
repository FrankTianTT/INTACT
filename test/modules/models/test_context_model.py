import torch

from intact.modules.models.context_model import ContextModel


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


def test_fix():
    from torch.optim import SGD

    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_context_dim = 2
    task_num = 5

    context_model = ContextModel(meta=True, max_context_dim=max_context_dim, task_num=task_num).to(device)
    context_model.set_context(torch.ones_like(context_model.context_hat) * 0.25)
    context_model.fix([1])
    optim = SGD(context_model.parameters(), lr=0.1)

    for i in range(100):
        context_model.zero_grad()
        loss = (context_model.context_hat**2).sum()
        loss.backward()
        optim.step()

    assert (context_model.context_hat[:, 0] < 1e-5).all()
    assert torch.allclose(context_model.context_hat[:, 1], torch.tensor(0.25))

    context_model.unfix()

    for i in range(100):
        context_model.zero_grad()
        loss = (context_model.context_hat**2).sum()
        loss.backward()
        optim.step()

    assert (context_model.context_hat < 1e-5).all()
