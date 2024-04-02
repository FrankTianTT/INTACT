import torch

from intact.stats.metric import mutual_info_estimation


def test_mutual_info_estimation():
    from torch.distributions import MultivariateNormal

    torch.manual_seed(0)

    batch_size = 100
    dim = 5
    test_num = 10

    def single_test(independent=True):
        loc = torch.randn(dim)
        if independent:
            covariance_matrix = torch.diag_embed(torch.randn(dim).abs())
        else:
            matrix = torch.randn(dim, dim)
            covariance_matrix = torch.mm(matrix, matrix.t())

        dist = MultivariateNormal(loc, covariance_matrix)
        values = dist.sample((batch_size,))

        mi = mutual_info_estimation(values, kernel_type="gaussian", reduction="mean")
        return mi

    independent_result = [single_test(True) for i in range(test_num)]
    dependent_result = [single_test(False) for i in range(test_num)]
    print("independent: ", sum(independent_result) / test_num)
    print("dependent: ", sum(dependent_result) / test_num)
