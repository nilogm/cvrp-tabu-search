from copy import deepcopy
from cvrp_tabu_search.utils import get_route_demand
from cvrp_tabu_search.problem import Solution, Instance


def clarke_wright(p: Instance) -> Solution:
    demand_points = [i for i in range(p.n) if i != p.depot_idx]
    routes = [[i] for i in demand_points]

    savings = [(p.w[p.depot_idx, i] + p.w[p.depot_idx, j] - p.w[i, j], [i, j]) for i in demand_points for j in demand_points[i + 1 :]]
    savings = sorted(savings, key=lambda x: x[0], reverse=True)

    def fits(r1: list[int], r2: list[int]) -> bool:
        return get_route_demand(r1, p.d) + get_route_demand(r2, p.d) <= p.c

    def get_route(vertex: int) -> list[int]:
        r = routes[vertex]
        while isinstance(r, int):
            r = routes[r]
        return r

    def merge_routes(r1: list, r2: list):
        r1.extend(deepcopy(r2))
        for l in r2:
            routes[l - 1] = routes.index(r1)

    for k in savings:
        i, j = k[1]
        r1 = get_route(i - 1)
        r2 = get_route(j - 1)

        if r1 == r2:
            continue

        if i == r1[0] and r2[-1] == j and fits(r2, r1):
            merge_routes(r2, r1)
        elif i == r1[-1] and j == r2[0] and fits(r1, r2):
            merge_routes(r1, r2)

    return Solution([i for i in routes if isinstance(i, list)], p.d, p.w)
