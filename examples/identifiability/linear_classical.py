import numpy as np
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import seaborn as sns

from causal_meta.stats.mcc import mean_corr_coef
from examples.identifiability.data_ganeration import gen_linear_data


def identify_theta(x, y, theta_size):
    task_num, sample_per_task, x_size = x.shape
    *_, y_size = y.shape

    b = np.empty((task_num, y_size))
    w = np.empty((task_num, y_size, x_size))
    for task_id in range(task_num):
        # linear regression
        lr = LinearRegression()
        lr.fit(x[task_id], y[task_id])

        b[task_id] = lr.intercept_
        w[task_id] = lr.coef_

        # mse = np.mean((y[task_id] - x[task_id] @ lr.coef_.T - lr.intercept_) ** 2)
        # print(mse)
    assert (np.var(w, axis=0) < 1e-3).all()
    total_w = np.mean(w, axis=0)

    ica = FastICA(n_components=theta_size)
    theta_hat = ica.fit_transform(b)

    # get mse
    hat_b = theta_hat @ ica.mixing_.T + ica.mean_
    hat_y = np.einsum("tsx,tyx->tsy", x, w) + hat_b.reshape(task_num, 1, y_size)
    mse = np.mean((hat_y - y) ** 2)
    print(mse)

    sns.displot(theta_hat[:, 0], kde=True, bins=20)
    plt.show()

    return theta_hat


if __name__ == '__main__':
    task_num = 1000
    sample_per_task = 100
    x_size = 5
    y_size = 8
    theta_size = 4
    seed = 0

    np.random.seed(seed)

    x, y, theta, A, B = gen_linear_data(task_num, sample_per_task, x_size, y_size, theta_size)
    theta_hat = identify_theta(x, y, theta_size)
    print(mean_corr_coef(theta_hat, theta))
    # 0.9993887945704704

    # x, y, theta, *_ = gen_linear_data(task_num, sample_per_task, x_size, y_size, theta_size, theta_is_gaussian=True)
    # theta_hat = identify_theta(x, y, theta_size)
    # print(mean_corr_coef(theta_hat, theta))
    # # 0.9108580802750957
