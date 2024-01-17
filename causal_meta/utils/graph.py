import numpy as np


def check_structural_sparsity(graph):
    for i in range(graph.shape[1]):
        lines = graph[graph[:, i] == 1]
        if len(lines) == 0:
            return False
        idxes = [set(np.where(line == 1)[0]) for line in lines]
        intersection = idxes[0].intersection(*idxes[1:])
        if len(intersection) != 1:
            return False
    return True
