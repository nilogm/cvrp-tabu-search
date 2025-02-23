import vrplib
import numpy as np
import pandas as pd
from math import log10
from cvrp_tabu_search.utils import get_route_demand, objective_function


class Solution:
    def __init__(self, s: list[int], d: np.ndarray, w: np.ndarray, f: int = None):
        self.s: list[list[int]] = s
        self.d: list[int] = [get_route_demand(r, d) for r in s]
        self.f: int = f if f else objective_function(s, w)

    def __str__(self):
        return str(self.f) + "  -  " + str(self.s)


class Instance:
    def __init__(self):
        self.name: str
        self.w: np.ndarray
        self.d: np.ndarray
        self.c: int
        self.depot_idx: int
        self.n: int
        self.solution: dict


class Run:
    def __init__(self, s: Solution, n: int, tabu_tenure_multiplier: float, bias_multiplier: float, seed: int):
        self.common_movements: dict[int, int] = {i: 0 for i in range(n)}
        self.common_movements_avg: float = 0
        self.tabu_list: list[int] = []
        self.tabu_tenures: list[int] = []
        self.initial_tabu_tenure: int = round(tabu_tenure_multiplier * log10(n))
        self.tabu_tenure_value: int = round(tabu_tenure_multiplier * log10(n))
        self.bias_multiplier: float = bias_multiplier
        self.best_solution: Solution = s
        self.savefile_suffix = f"t_{tabu_tenure_multiplier}_m_{bias_multiplier}_s_{seed}.csv"
        self.savefile = pd.DataFrame()
        self.seed: int = seed

    def begin_savefile(self, file_save_path: str, instance_name: str):
        self.save_path = f"{file_save_path}/{instance_name}__{self.savefile_suffix}"

    def update_savefile(self, s: Solution, is_restart: bool, time: float):
        self.savefile = pd.concat([self.savefile, pd.DataFrame({"local": [s.f], "global": [self.best_solution.f], "restart": [is_restart], "time": [time]})], ignore_index=True)

    def save(self):
        self.savefile.to_csv(self.save_path)


def get_instance(path: str) -> Instance:
    instance = vrplib.read_instance(f"{path}.vrp")

    p = Instance()
    p.name = instance["name"]
    p.w = np.round(instance["edge_weight"])
    p.d = instance["demand"]
    p.c = instance["capacity"]
    p.depot_idx = instance["depot"]
    p.n = instance["dimension"]

    p.solution = vrplib.read_solution(f"{path}.sol")

    return p
