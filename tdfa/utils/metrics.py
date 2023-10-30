import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


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
    :param return_permutation: bool, optional
    :return: float
    """
    
    d = x.shape[1]
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
    elif method == 'spearman':
        cc = spearmanr(x, y)[0][:d, d:]
    else:
        raise ValueError('not a valid method: {}'.format(method))
    cc = np.abs(cc)
    permutation = linear_sum_assignment(-1 * cc)
    score = cc[permutation].mean()
    if return_permutation:
        return score, permutation
    else:
        return score
