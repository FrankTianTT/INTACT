import numpy as np


def check_structural_sparsity(graph):
    """
    Check the structural sparsity of a given graph.

    This function checks if the graph is structurally sparse. A graph is considered structurally sparse
    if for each column in the adjacency matrix, all rows with a value of 1 have exactly one common index
    where the value is 1.

    Args:
        graph (np.ndarray): The adjacency matrix of the graph.

    Returns:
        bool: True if the graph is structurally sparse, False otherwise.
    """
    for i in range(graph.shape[1]):
        lines = graph[graph[:, i] == 1]
        if len(lines) == 0:
            return False
        idxes = [set(np.where(line == 1)[0]) for line in lines]
        intersection = idxes[0].intersection(*idxes[1:])
        if len(intersection) != 1:
            return False
    return True
