import os
import json
import typer
import traceback
from itertools import product
from typing_extensions import Annotated
from cvrp_tabu_search.problem import get_instance
from cvrp_tabu_search.tabu_search import run_tabu

app_experiment = typer.Typer()


def _exec_run():
    app_experiment()


def init(config_file: str, results_file: str):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file could not be found! File path: '{config_file}'")

    with open(config_file) as f:
        d = json.load(f)

    if not ("tabu_tenure_values" in d and "bias_multiplier_values" in d and "instances" in d and "run_time" in d and "seeds" in d):
        raise AttributeError(
            "Configuration file has missing values. Make sure it contains at least the following keys: 'tabu_tenure_values', 'bias_multiplier_values', 'instances', 'run_time' and 'seeds'"
        )

    all_configs = [i for i in product(d["tabu_tenure_values"], d["bias_multiplier_values"], d["seeds"])]

    results_file_path = os.path.join(os.getcwd(), results_file)
    os.makedirs(results_file_path, exist_ok=True)

    return d, all_configs


def check_instance_path(instance_path: str):
    path = os.path.join(os.getcwd(), instance_path)

    if os.path.isdir(path):
        return path

    if not os.path.exists(f"{path}.vrp"):
        print(f"Instance file could not be found! File path: '{path}.vrp'")
        return None

    if not os.path.exists(f"{path}.sol"):
        print(f"Instance file could not be found! File path: '{path}.sol'")
        return None

    return path


def run(instance_path: str, run_time: int, all_configs: list, results_file: str):
    instance = get_instance(instance_path)
    for tabu_tenure, bias_multiplier, seed in all_configs:
        print(instance_path, tabu_tenure, bias_multiplier, seed)
        run_tabu(instance, run_time, tabu_tenure, bias_multiplier, seed, results_file)


@app_experiment.command(help="Executes the experiments")
def exec(
    config_file: Annotated[str, typer.Option(help="Configuration file for the run")],
    results_file: Annotated[str, typer.Option(help="Directory in which to save the run's .csv")],
):
    c, all_configs = init(config_file, results_file)

    for instance_path in c["instances"]:
        path = check_instance_path(instance_path)
        if path is None:
            continue

        if os.path.isdir(path):
            instances = set([check_instance_path(os.path.join(path, i.removesuffix(".vrp"))) for i in os.listdir(path) if i.endswith(".vrp")])
            for i in instances:
                path_ = os.path.join(os.getcwd(), i)
                try:
                    run(path_, c["run_time"], all_configs, results_file)
                except Exception as e:
                    print(e)
                    print(traceback.format_exc(e))
        else:
            try:
                run(path, c["run_time"], all_configs, results_file)
            except Exception as e:
                print(e)
                print(traceback.format_exc(e))
