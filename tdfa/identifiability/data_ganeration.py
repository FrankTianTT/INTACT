import numpy as np


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
