import numpy as np
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression

from tdfa.utils.metrics import mean_corr_coef


def date_generation(task_num, sample_per_task, x_size, y_size, theta_size, theta_is_gaussian=False):
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


def seed_everything(seed=0):
    np.random.seed(seed)


def identify_theta(x, y):
    b = np.empty((task_num, y_size))
    for task_id in range(task_num):
        # linear regression
        lr = LinearRegression()
        lr.fit(x[task_id], y[task_id])

        b[task_id] = lr.intercept_

    ica = FastICA(n_components=theta_size)
    components = ica.fit_transform(b)

    return components


if __name__ == '__main__':
    seed_everything()

    task_num = 1000
    sample_per_task = 100
    x_size = 5
    y_size = 8
    theta_size = 3

    x, y, theta, A, B = date_generation(task_num, sample_per_task, x_size, y_size, theta_size)
    theta_hat = identify_theta(x, y)
    print(mean_corr_coef(theta_hat, theta))
    # 0.9993887945704704

    x, y, theta, A, B = date_generation(task_num, sample_per_task, x_size, y_size, theta_size, theta_is_gaussian=True)
    theta_hat = identify_theta(x, y)
    theta_hat = identify_theta(x, y)
    print(mean_corr_coef(theta_hat, theta))
    # 0.9097250263456617
