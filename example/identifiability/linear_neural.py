import numpy as np
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
import seaborn as sns

from tdfa.utils.metrics import mean_corr_coef
from example.identifiability.data_ganeration import gen_linear_data
from tdfa.stats.metric import mutual_info_estimation


def identify_theta(x, y, theta_size):
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    task_num, sample_per_task, x_size = x.shape
    *_, y_size = y.shape

    model = torch.nn.Linear(x_size + theta_size, y_size, bias=False)
    theta_hat = torch.nn.Parameter(torch.randn(task_num, theta_size))

    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    theta_optimizer = torch.optim.Adam([theta_hat], lr=0.1)

    mse_list, mi_list = [], []
    for epoch in range(100):
        x_theta = torch.cat([x, theta_hat.unsqueeze(1).expand(-1, sample_per_task, -1)], dim=-1)
        y_hat = model(x_theta)

        reg = mutual_info_estimation(theta_hat) * 5
        mse = F.mse_loss(y_hat, y)
        loss = mse + reg

        mi_list.append(reg.item())
        mse_list.append(mse.item())

        model_optimizer.zero_grad()
        theta_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()
        theta_optimizer.step()

    print(mse_list[-1], mi_list[-1])
    plt.plot(mse_list, label='mse')
    plt.plot(mi_list, label='reg')
    plt.legend()
    plt.show()

    sns.displot(theta_hat.detach().numpy()[:, 0], kde=True, bins=20)
    plt.show()

    return theta_hat.detach().numpy()


if __name__ == '__main__':
    task_num = 100
    sample_per_task = 10
    x_size = 5
    y_size = 8
    theta_size = 4
    seed = 0

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    x, y, theta, *_ = gen_linear_data(task_num, sample_per_task, x_size, y_size, theta_size)
    theta_hat = identify_theta(x, y, theta_size)
    print(mean_corr_coef(theta_hat, theta))
    # 0.7452454792009036

    # x, y, theta, *_ = gen_linear_data(task_num, sample_per_task, x_size, y_size, theta_size, theta_is_gaussian=True)
    # theta_hat = identify_theta(x, y, theta_size)
    # print(mean_corr_coef(theta_hat, theta))
    # # 0.8411799184231902

    # theta = torch.distributions.Laplace(0, 1).sample((task_num, theta_size))
    # mutual_info_estimation(theta)

    # 0.5751473268914185
    # 0.36141771776103054
