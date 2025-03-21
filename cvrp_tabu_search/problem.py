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

    def get_overcapacity(self, max_c: int):
        return sum([max(0, i - max_c) for i in self.d])

    def __str__(self):
        return str(self.s)

    def __len__(self):
        return ([len(i) > 0 for i in self.s]).count(True)

    def min(self):
        return min([len(i) for i in self.s])


class Instance:
    def __init__(self):
        self.name: str
        self.w: np.ndarray
        self.d: np.ndarray
        self.c: int
        self.depot_idx: int
        self.n: int
        self.k: int
        self.solution: dict


class Parameters:
    def __init__(self, n: int, tabu_tenure_multiplier: float, frequency_multiplier: float, invalid_multiplier: float):
        self.tabu_tenure = tabu_tenure_multiplier
        self.t: int = round(tabu_tenure_multiplier * log10(n))
        self.f: float = frequency_multiplier
        self.i: float = invalid_multiplier


class Run:
    def __init__(self, s: Solution, n: int, valid_parameters: Parameters, invalid_parameters: Parameters, seed: int = None):
        self.common_movements: dict[int, int] = {i: 0 for i in range(n)}
        self.tabu_list = {i: [] for i in range(n)}
        self.tabu_tenures = {i: [] for i in range(n)}

        self.params = invalid_parameters
        self.valid_parameters = valid_parameters
        self.invalid_parameters = invalid_parameters
        self.invalid_mode = True

        self.a = 1
        self.b = 1

        self.best_solution: Solution = s
        self.savefile_suffix = (
            f"t_{valid_parameters.tabu_tenure}_f_{valid_parameters.f}_o_{valid_parameters.i}_t_{invalid_parameters.tabu_tenure}_f_{invalid_parameters.f}_o_{invalid_parameters.i}_s_{seed}.csv"
        )
        self.savefile = pd.DataFrame({"local": [s.f], "global": [s.f], "time": [0.0], "solution": [s.s]})
        self.seed: int = seed

    def begin_savefile(self, file_save_path: str, instance_name: str):
        self.save_path = f"{file_save_path}/{instance_name}__{self.savefile_suffix}"

    def update_savefile(self, s: Solution, time: float, over_k: bool, over_c: bool):
        self.savefile = pd.concat(
            [self.savefile, pd.DataFrame({"local": [s.f], "global": [self.best_solution.f], "time": [time], "solution": [s.s], "over_k": [over_k], "over_c": [over_c]})], ignore_index=True
        )

    def save(self):
        self.savefile.to_csv(self.save_path)

    def reset_values(self):
        if self.invalid_mode:
            self.params = self.valid_parameters
            self.invalid_mode = False


def get_instance(path: str) -> Instance:
    instance = vrplib.read_instance(f"{path}.vrp")

    p = Instance()
    p.name = instance["name"]
    p.w = np.round(instance["edge_weight"])
    p.d = instance["demand"]
    p.c = instance["capacity"]
    p.depot_idx = instance["depot"]
    p.n = instance["dimension"]
    p.k = int(p.name.split("k")[-1])

    p.solution = vrplib.read_solution(f"{path}.sol")

    return p
