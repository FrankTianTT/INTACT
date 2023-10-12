import abc

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

from tdfa.utils.metrics import mean_corr_coef
from tdfa.identifiability.data_ganeration import gen_linear_data


class Kernel(abc.ABC, nn.Module):
    """Base class which defines the interface for all kernels."""

    def __init__(self, bandwidth=1.0):
        """Initializes a new Kernel.

        Args:
            bandwidth: The kernel's (band)width.
        """
        super().__init__()
        self.bandwidth = bandwidth

    def _diffs(self, test_Xs, train_Xs):
        """Computes difference between each x in test_Xs with all train_Xs."""
        test_Xs = test_Xs.view(test_Xs.shape[0], 1, *test_Xs.shape[1:])
        train_Xs = train_Xs.view(1, train_Xs.shape[0], *train_Xs.shape[1:])
        return test_Xs - train_Xs

    @abc.abstractmethod
    def forward(self, test_Xs, train_Xs):
        """Computes log p(x) for each x in test_Xs given train_Xs."""

    @abc.abstractmethod
    def sample(self, train_Xs):
        """Generates samples from the kernel distribution."""


class ParzenWindowKernel(Kernel):
    """Implementation of the Parzen window kernel."""

    def forward(self, test_Xs, train_Xs):
        abs_diffs = torch.abs(self._diffs(test_Xs, train_Xs))
        dims = tuple(range(len(abs_diffs.shape))[2:])
        dim = np.prod(abs_diffs.shape[2:])
        inside = torch.sum(abs_diffs / self.bandwidth <= 0.5, dim=dims) == dim
        coef = 1 / self.bandwidth ** dim
        return torch.log((coef * inside).mean(dim=1))

    @torch.no_grad()
    def sample(self, train_Xs):
        device = train_Xs.device
        noise = (torch.rand(train_Xs.shape, device=device) - 0.5) * self.bandwidth
        return train_Xs + noise


class GaussianKernel(Kernel):
    """Implementation of the Gaussian kernel."""

    def forward(self, test_Xs, train_Xs):
        n, d = train_Xs.shape
        n, h = torch.tensor(n, dtype=torch.float32), torch.tensor(self.bandwidth)
        pi = torch.tensor(np.pi)

        Z = 0.5 * d * torch.log(2 * pi) + d * torch.log(h) + torch.log(n)
        diffs = self._diffs(test_Xs, train_Xs) / h
        log_exp = -0.5 * torch.norm(diffs, p=2, dim=-1) ** 2

        return torch.logsumexp(log_exp - Z, dim=-1)

    @torch.no_grad()
    def sample(self, train_Xs):
        device = train_Xs.device
        noise = torch.randn(train_Xs.shape, device=device) * self.bandwidth
        return train_Xs + noise


def mutual_info_estimation(values, kernel_type="gaussian"):
    # values: batch * dim
    kernel_class = {"gaussian": GaussianKernel, "parzen": ParzenWindowKernel}[kernel_type]

    kernel = kernel_class()
    log_pdf1 = kernel(values[:, 0:1], values[:, 0:1])
    entropy1 = - torch.sum(log_pdf1 * torch.exp(log_pdf1))
    log_pdf2 = kernel(values[:, 1:2], values[:, 1:2])
    entropy2 = - torch.sum(log_pdf2 * torch.exp(log_pdf2))

    log_pdf12 = kernel(values, values)
    entropy12 = - torch.sum(log_pdf12 * torch.exp(log_pdf12))

    mi = entropy1 + entropy2 - entropy12
    return mi


def identify_theta(x, y, theta_size):
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    task_num, sample_per_task, x_size = x.shape
    *_, y_size = y.shape

    model = torch.nn.Linear(x_size + theta_size, y_size, bias=False)
    theta_hat = torch.nn.Parameter(torch.randn(task_num, theta_size))

    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    theta_optimizer = torch.optim.SGD([theta_hat], lr=0.1)

    mse_list, mi_list = [], []
    for epoch in range(1000):
        x_theta = torch.cat([x, theta_hat.unsqueeze(1).expand(-1, sample_per_task, -1)], dim=-1)
        y_hat = model(x_theta)

        mi = mutual_info_estimation(theta_hat)
        mse = F.mse_loss(y_hat, y)
        mi_list.append(mi.item())
        mse_list.append(mse.item())
        loss = mse + mi

        model_optimizer.zero_grad()
        theta_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()
        theta_optimizer.step()

    plt.plot(mse_list, label='mse')
    plt.plot(mi_list, label='mi')
    plt.legend()
    plt.show()
    return theta_hat.detach().numpy()


if __name__ == '__main__':
    task_num = 100
    sample_per_task = 10
    x_size = 5
    y_size = 8
    theta_size = 2
    seed = 1

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    x, y, theta, *_ = gen_linear_data(task_num, sample_per_task, x_size, y_size, theta_size)
    theta_hat = identify_theta(x, y, theta_size)
    print(mean_corr_coef(theta_hat, theta))
    # 0.7452454792009036

    x, y, theta, *_ = gen_linear_data(task_num, sample_per_task, x_size, y_size, theta_size, theta_is_gaussian=True)
    theta_hat = identify_theta(x, y, theta_size)
    print(mean_corr_coef(theta_hat, theta))
    # 0.8411799184231902

    # theta = torch.distributions.Laplace(0, 1).sample((task_num, theta_size))
    # mutual_info_estimation(theta)
