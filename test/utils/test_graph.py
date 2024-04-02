import numpy as np

from intact.utils.graph import check_structural_sparsity


def test_check_structural_sparsity1():
    graph = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert not check_structural_sparsity(graph)


def test_check_structural_sparsity2():
    graph = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    assert check_structural_sparsity(graph)
