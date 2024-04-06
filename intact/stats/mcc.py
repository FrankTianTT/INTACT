import numpy as np
from causallearn.utils.KCI.KCI import KCI_UInd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score


def kernel_independence_test(x, y):
    x_dim, y_dim = x.shape[1], y.shape[1]

    ci_test = KCI_UInd(approx=False)

    result_matrix = np.zeros((x_dim, y_dim))
    for i in range(x_dim):
        for j in range(y_dim):
            p_value, test_stat = ci_test.compute_pvalue(
                x[:, i : i + 1], y[:, j : j + 1]
            )
            result_matrix[i, j] = 1 - min(0.05, p_value) * 20
    return result_matrix


def kernel_ridge_regression(x, y):
    x_dim, y_dim = x.shape[1], y.shape[1]

    result_matrix = np.zeros((x_dim, y_dim))
    for i in range(x_dim):
        for j in range(y_dim):
            krr = KernelRidge(alpha=1.0, kernel="rbf", gamma=3.0)
            krr.fit(x[:, i : i + 1], y[:, j])
            y_hat = krr.predict(x[:, i : i + 1])
            r2 = r2_score(y[:, j], y_hat)

            result_matrix[i, j] = r2
    return result_matrix


def mean_corr_coef(x, y, method="pearson", return_permutation=False):
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
    if method == "pearson":
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
        cc = np.abs(cc)
    elif method == "spearman":
        cc = spearmanr(x, y)[0][:d, d:]
        cc = np.abs(cc)
    elif method == "kit":
        cc = kernel_independence_test(x, y)
    elif method == "krr":
        cc = kernel_ridge_regression(x, y)
    else:
        raise ValueError("not a valid method: {}".format(method))

    if np.isnan(cc).any():
        cc = np.nan_to_num(cc)

    permutation = linear_sum_assignment(-1 * cc)

    if len(permutation[0]) == 0:
        score = 0
    else:
        # score = cc[permutation].sum() / max(x.shape[1], y.shape[1])
        score = cc[permutation].mean()

    if return_permutation:
        return score, permutation
    else:
        return score
