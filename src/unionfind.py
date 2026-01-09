# class UnionFind():
#     representatives: list[int]
#     def __init__(self, N: int) -> None:
#         self.representatives = list(range(N))
#     def find_representative(self, i: int) -> int:
#         parent: int = self.representatives[i]
#         if parent == i:
#             return i
#         return self.find_representative(self.representatives[i])

from numba import njit
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
