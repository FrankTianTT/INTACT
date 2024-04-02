import numpy as np

from causal_meta.stats.mcc import mean_corr_coef


def test_non_linear():
    import matplotlib.pyplot as plt

    sample_num = 100

    x = (np.random.random([sample_num, 1]) - 0.5) * 2
    y = np.abs(x)  # + np.random.random([sample_num, 1]) * 0.2
    # plt.scatter(x, y)
    # plt.show()

    mcc1 = mean_corr_coef(x, y, "krr")
    mcc2 = mean_corr_coef(x, y, "pearson")

    print(mcc1, mcc2)


def test_mcc():
    sample_num = 100

    x = np.random.random([sample_num, 3])
    y = np.random.random([sample_num, 3])

    mcc = mean_corr_coef(x, y)
    print(mcc)
