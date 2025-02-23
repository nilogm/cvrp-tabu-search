import numpy as np


def objective_function(s: list[list[int]], w: np.ndarray) -> int:
    f = 0
    for r in s:
        f += w[0, r[0]] + w[0, r[-1]]
        if len(r) > 1:
            for v, u in zip(r, r[1:]):
                f += w[v, u]
    return f


def prev_vertex(r: list[int], i: int):
    if i == 0:
        return 0
    return r[i - 1]


def next_vertex(r: list[int], i: int):
    if i == len(r) - 1:
        return 0
    return r[i + 1]


def get_route_demand(r: list[int], d: np.ndarray):
    if len(r) == 0:
        return 0
    return sum([d[i] for i in r])
