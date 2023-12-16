import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, gaussian_kde, differential_entropy
from sklearn.feature_selection import mutual_info_regression
from causallearn.utils.KCI.KCI import KCI_UInd


def kernel_independence_test(x, y):
    x_dim, y_dim = x.shape[1], y.shape[1]

    ci_test = KCI_UInd(approx=False)

    nmi_matrix = np.zeros((x_dim, y_dim))
    for i in range(x_dim):
        for j in range(y_dim):
            p_value, test_stat = ci_test.compute_pvalue(x[:, i:i + 1], y[:, j:j + 1])
            nmi_matrix[i, j] = 1 - min(0.05, p_value) * 20

    return nmi_matrix


def mean_corr_coef(x, y, method='pearson', return_permutation=False):
    """
    A numpy implementation of the mean correlation coefficient metric.

    :param x: numpy.ndarray
    :param y: numpy.ndarray
    :param method: str, optional
            The method used to compute the correlation coefficients.
                The options are 'pearson' and 'spearman'
                'pearson':
                    use Pearson's correlation coefficient
                'spearman':
                    use Spearman's nonparametric rank correlation coefficient
                'kernel':
                    use KCI
    :param return_permutation: bool, optional
    :return: float
    """

    d = x.shape[1]
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
        cc = np.abs(cc)
    elif method == 'spearman':
        cc = spearmanr(x, y)[0][:d, d:]
        cc = np.abs(cc)
    elif method == 'kernel':
        cc = kernel_independence_test(x, y)
    else:
        raise ValueError('not a valid method: {}'.format(method))

    if np.isnan(cc).any():
        cc = np.nan_to_num(cc)

    permutation = linear_sum_assignment(-1 * cc)

    if len(permutation[0]) == 0:
        score = 0
    else:
        score = cc[permutation].mean()

    if return_permutation:
        return score, permutation
    else:
        return score


def test_non_linear():
    import matplotlib.pyplot as plt

    sample_num = 1000

    x = (np.random.random([sample_num, 1]) - 0.5) * 2
    y = np.abs(x) + np.random.random([sample_num, 1]) * 0.2
    plt.scatter(x, y)
    plt.show()

    mcc1 = mean_corr_coef(x, y, "kernel")
    mcc2 = mean_corr_coef(x, y, "pearson")

    print(mcc1, mcc2)


def test_mcc():
    sample_num = 100

    x = np.random.random([sample_num, 1])
    y = np.random.random([sample_num, 0])

    mcc = mean_corr_coef(x, y, "kernel")


if __name__ == '__main__':
    test_mcc()
