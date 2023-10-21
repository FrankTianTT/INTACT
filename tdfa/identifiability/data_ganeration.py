import numpy as np
import torch

from tdfa.models.util import build_parallel_layers


def gen_linear_data(task_num, sample_per_task, x_size, y_size, theta_size, theta_is_gaussian=False):
    # y = xA^T + \thetaB^T + \epsilon

    # static parameters
    A = np.random.randn(y_size, x_size)
    B = np.random.randn(y_size, theta_size)

    # theta and x
    if theta_is_gaussian:
        theta = np.random.randn(task_num, theta_size)
    else:
        theta = np.random.laplace(0, 1, (task_num, theta_size))
    x = np.random.randn(task_num, sample_per_task, x_size)

    # y
    y = x @ A.T + (theta @ B.T).reshape(task_num, 1, y_size)
    y += np.random.randn(task_num, sample_per_task, y_size) * 0.1

    return x, y, theta, A, B


def gen_causal_graph(x_size, y_size, theta_size, sparse_p=0.5):
    graph = np.random.choice([0, 1], size=(y_size, x_size + theta_size), p=[sparse_p, 1 - sparse_p])

    while not is_sparse(graph):
        graph = np.random.choice([0, 1], size=(y_size, x_size + theta_size), p=[sparse_p, 1 - sparse_p])

    return graph


def is_sparse(graph):
    for i in range(graph.shape[1]):
        lines = graph[graph[:, i] == 1]
        if len(lines) == 0:
            return False
        idxes = [set(np.where(line == 1)[0]) for line in lines]
        intersection = idxes[0].intersection(*idxes[1:])
        if len(intersection) != 1:
            return False
    return True


def gen_nonlinear_data(task_num, sample_per_task, x_size, y_size, theta_size, theta_is_gaussian=False):
    model = build_parallel_layers(x_size + theta_size, 1, [32], extra_dims=[y_size], activate_name='Tanh')

    for name, p in model.named_parameters():
        if 'weight' in name:
            p.data = torch.randn_like(p.data)

    x = torch.randn(task_num, sample_per_task, x_size)
    theta = torch.distributions.Laplace(0, 1).sample((task_num, theta_size))
    x_theta = torch.cat([x, theta.unsqueeze(1).expand(-1, sample_per_task, -1)], dim=-1)

    graph = gen_causal_graph(x_size, y_size, theta_size)  # out-size * in-size
    mask = torch.from_numpy(graph).float()
    repeated_x_theta = x_theta.unsqueeze(0).expand(y_size, -1, -1, -1).reshape(y_size, -1, x_size + theta_size)
    masked_x_theta = torch.einsum("oi,obi->obi", mask, repeated_x_theta)

    y = model(masked_x_theta).permute(2, 1, 0)[0].reshape(task_num, sample_per_task, y_size)
    return x.detach().numpy(), y.detach().numpy(), theta.detach().numpy(), graph


if __name__ == '__main__':
    np.random.seed(0)
    torch.random.manual_seed(0)

    gen_nonlinear_data(100, 10, 4, 5, 3)
