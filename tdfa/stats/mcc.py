import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, gaussian_kde


def nmi_estimation(x, y):
    """Estimate the normalized mutual information between x and y.

    :param x: shape: (samples_num, x_dim)
    :param y: shape: (samples_num, y_dim)
    :return: normalized mutual information matrix, shape: (x_dim, y_dim)
    """

    # x, y = x / np.std(x, axis=0) - np.mean(x, axis=0), y / np.std(y, axis=0) - np.mean(y, axis=0)
    x_dim, y_dim = x.shape[1], y.shape[1]

    log_pdf_x = np.stack([gaussian_kde(x[:, i]).logpdf(x[:, i]) for i in range(x_dim)])
    log_pdf_y = np.stack([gaussian_kde(y[:, j]).logpdf(y[:, j]) for j in range(y_dim)])
    hx = -np.mean(log_pdf_x, axis=-1)
    hy = -np.mean(log_pdf_y, axis=-1)

    nmi_matrix = np.zeros((x_dim, y_dim))
    for i in range(x_dim):
        for j in range(y_dim):
            xy = np.stack([x[:, i], y[:, j]], axis=0)
            log_pdf_ij = gaussian_kde(xy).logpdf(xy)
            mutual_info_ij = (log_pdf_ij - log_pdf_x[i] - log_pdf_y[j]).mean()
            print(mutual_info_ij, hx[i], hy[j])
            nmi_matrix[i, j] = mutual_info_ij / np.sqrt(hx[i] * hy[j])

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
                'nmi':
                    use normalized mutual information
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
    elif method == 'nmi':  # normalized mutual information
        cc = nmi_estimation(x, y)
    else:
        raise ValueError('not a valid method: {}'.format(method))

    if np.isnan(cc).any():
        cc = np.nan_to_num(cc)

    permutation = linear_sum_assignment(-1 * cc)

    score = cc[permutation].mean()
    if return_permutation:
        return score, permutation
    else:
        return score


def test_non_linear():
    sample_num = 1000

    x = np.random.randn(sample_num, 1)
    y = np.abs(x)

    mcc1 = mean_corr_coef(x, y, "nmi")
    mcc2 = mean_corr_coef(x, y, "pearson")

    print(mcc1, mcc2)


if __name__ == '__main__':
    # test_non_linear()
    print(np.sqrt(-1))