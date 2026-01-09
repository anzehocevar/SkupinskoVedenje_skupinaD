from numba import njit
import numpy as np
import numpy.typing as npt

@njit
def find(representatives: npt.NDArray, i: int) -> int:
    i_original: int = i
    parent: int = representatives[i]
    while parent != i:
        i = parent
        parent = representatives[i]
    root: int = parent

    i = i_original
    while representatives[i] != root:
        parent = representatives[i]
        representatives[i] = root
        i = parent

    return root

@njit
def union(representatives: npt.NDArray, i: int, j: int):
    rep_i: int = find(representatives, i)
    rep_j: int = find(representatives, j)
    if rep_i == rep_j:
        return
    representatives[rep_i] = rep_j

@njit
def compress_all(representatives: npt.NDArray):
    for i in range(representatives.shape[0]):
        representatives[i] = find(representatives, i)

@njit
def sort_by_frequency(group):
    uniques: list[int] = []
    for x in group:
        if x not in uniques:
            uniques.append(x)

    counts: npt.NDArray = np.zeros(max(uniques)+1, dtype=np.int64)
    for x in group:
        counts[x] += 1
    if np.all(counts == 0):
        return np.empty(0, dtype=np.int64)

    order: npt.NDArray = np.argsort(counts)
    first_non_zero: int = 0
    while counts[order[first_non_zero]] < 1:
        first_non_zero += 1
    order = order[first_non_zero:]
    order = order[::-1]

    return order
