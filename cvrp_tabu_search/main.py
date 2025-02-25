import os
import json
import typer
import traceback
from itertools import product
from typing_extensions import Annotated
from cvrp_tabu_search.problem import get_instance
from cvrp_tabu_search.tabu_search import run_tabu
from cvrp_tabu_search.utils import objective_function
import pandas as pd
import matplotlib.pyplot as plt

app_experiment = typer.Typer()


def _exec_run():
    app_experiment()


def init(config_file: str, results_folder: str):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file could not be found! File path: '{config_file}'")

    with open(config_file) as f:
        d = json.load(f)

    if not ("tabu_tenure_values" in d and "bias_multiplier_values" in d and "instances" in d and "run_time" in d and "seeds" in d):
        raise AttributeError(
            "Configuration file has missing values. Make sure it contains at least the following keys: 'tabu_tenure_values', 'bias_multiplier_values', 'instances', 'run_time' and 'seeds'"
        )

    all_configs = [i for i in product(d["tabu_tenure_values"], d["bias_multiplier_values"], d["seeds"])]

    results_folder_path = os.path.join(os.getcwd(), results_folder)
    os.makedirs(results_folder_path, exist_ok=True)

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


def run(instance_path: str, run_time: int, all_configs: list, results_folder: str):
    """Executa o algoritmo para a instância dada.

    Args:
        instance_path (str): caminho para a instância
        run_time (int): tempo de execução
        all_configs (list): todas as combinações de parâmetros para testar
        results_folder (str): diretório de destino dos resultados
    """
    instance = get_instance(instance_path)
    for tabu_tenure, bias_multiplier, seed in all_configs:
        print(instance_path, tabu_tenure, bias_multiplier, seed)
        run_tabu(instance, run_time, tabu_tenure, bias_multiplier, seed, results_folder)


@app_experiment.command(help="Executes the experiments")
def exec(
    config_file: Annotated[str, typer.Option(help="Configuration file for the run")],
    results_folder: Annotated[str, typer.Option(help="Directory in which to save the run's .csv")],
):
    """Executa os experimentos descritos no arquivo de configuração e manda os resultados para a pasta dada.

    Args:
        config_file (Annotated[str, typer.Option, optional): arquivo de configuração. Defaults to "Configuration file for the run")].
        results_folder (Annotated[str, typer.Option, optional): pasta destino para os resultados. Defaults to "Directory in which to save the run's .csv")].
    """
    # carrega as configurações, cria as pastas
    c, all_configs = init(config_file, results_folder)

    for instance_path in c["instances"]:
        path = check_instance_path(instance_path)
        if path is None:
            continue

        # se for diretório, executa todos os itens do diretório
        if os.path.isdir(path):
            instances = set([check_instance_path(os.path.join(path, i.removesuffix(".vrp"))) for i in os.listdir(path) if i.endswith(".vrp")])
            for i in instances:
                path_ = os.path.join(os.getcwd(), i)
                try:
                    run(path_, c["run_time"], all_configs, results_folder)
                except Exception as e:
                    print(e)
                    print(traceback.format_exc(e))

        # se for um arquivo, executa a instância
        else:
            try:
                run(path, c["run_time"], all_configs, results_folder)
            except Exception as e:
                print(e)
                print(traceback.format_exc(e))


def load_instance(instance_name: str):
    if "A" in instance_name:
        letter = "A"
    elif "B" in instance_name:
        letter = "B"
    elif "F" in instance_name:
        letter = "F"

    instance_path = f"Vrp-Set-{letter}/{letter}/{instance_name}"
    return get_instance(os.path.join(os.getcwd(), instance_path))


@app_experiment.command(help="Plots the graphs")
def plot(result_file: Annotated[str, typer.Option(help="The target run's .csv")]):
    """Plota o gráfico de uma execução.

    Args:
        result_file (Annotated[str, typer.Option, optional): arquivo .csv com os resultados da execução. Defaults to "The target run's .csv")].
    """
    instance_name, _ = result_file.split("/")[-1].split("__")
    instance = load_instance(instance_name)

    df = pd.read_csv(result_file)

    fig, ax = plt.subplots(tight_layout=True)
    ax.scatter(df.index, df["local"], c="b", s=1, lw=1)
    ax.scatter(df.index, df["global"], c="g", s=1, lw=1)

    sol_cost = objective_function(instance.solution["routes"], instance.w)
    ax.axhline(sol_cost, c="r", lw=1)

    min_idx = df["global"].idxmin()
    ax.scatter([min_idx], [df.iloc[min_idx]["global"]], c="k", s=10, lw=1)

    ax.set_title(f"Instance: {instance_name} | Solution: {sol_cost} | Best: {df.iloc[min_idx]['global']} | Iteration: {min_idx}")
    ax.set_ylim(sol_cost - 10, df["local"].max() + 10)
    ax.set_xlim(-10, df.index[-1] + 10)

    fig.savefig("graph.svg")

    plt.show()


def read_folder(results_folder: Annotated[str, typer.Option(help="Directory containing results .csvs")]):
    folder_path = os.path.join(os.getcwd(), results_folder)
    files = sorted(os.listdir(folder_path))

    cols = ["Instance", "Solution", "Best", "Time", "Tenure", "Bias", "Gap"]
    all_df = pd.DataFrame()

    for i in files:
        instance_name, info = i.split("__")
        instance = load_instance(instance_name)

        df_path = os.path.join(folder_path, i)
        df = pd.read_csv(df_path)

        # sol_cost = objective_function(instance.solution["routes"], instance.w)
        sol_cost = instance.solution["cost"]

        _, tenure, _, bias, _, _ = info.split("_")

        best = df.iloc[-1]["global"]
        min_time = df.iloc[df["local"].idxmin()]["time"]
        gap = (best - sol_cost) / sol_cost

        all_df = pd.concat([all_df, pd.DataFrame([[instance_name, sol_cost, best, min_time, float(tenure), float(bias), gap]], columns=cols)])

    return all_df


@app_experiment.command(help="Shows the parameter tuning tables")
def analyze(results_folder: Annotated[str, typer.Option(help="Directory containing results .csvs")]):
    """Printa as tabelas dos resultados dos experimentos com os parâmetros para análise.

    Args:
        results_folder (Annotated[str, typer.Option, optional): caminho para o diretório. Defaults to "Directory containing results .csvs")].
    """
    all_df = read_folder(results_folder)

    analysis = all_df.groupby(["Tenure", "Bias"]).agg({"Gap": ["min", "mean", "std"]}).sort_values(["Tenure", "Bias"])
    print(analysis.to_latex())
    print()
    table = all_df.sort_values(["Instance", "Tenure", "Bias"])
    table["Solution"] = table["Solution"].astype(int)
    table["Best"] = table["Best"].astype(int)
    table["Time"] = table["Time"].map("{:.2f}".format)
    table["Tenure"] = table["Tenure"].map("{:.1f}".format)
    table["Bias"] = table["Bias"].map("{:.3f}".format)
    print(table.to_latex(index=False))
    print()
    
    print(analysis)
    print()
    print(table)


@app_experiment.command(help="Shows the final results table")
def table(results_folder: Annotated[str, typer.Option(help="Directory containing results .csvs")]):
    """Printa as tabelas dos resultados dos experimentos finais.

    Args:
        results_folder (Annotated[str, typer.Option, optional): caminho para o diretório. Defaults to "Directory containing results .csvs")].
    """
    all_df = read_folder(results_folder)
    all_df = all_df.drop(columns=["Tenure", "Bias"])

    table = all_df.sort_values(["Instance"]).groupby(["Instance", "Solution"], as_index=False).agg({"Best": ["min", "mean"], "Time": ["min", "mean"], "Gap": ["min", "mean"]})
    table["Solution"] = table["Solution"].astype(int)
    table["Best", "min"] = table["Best", "min"].astype(int)
    table["Best", "mean"] = table["Best", "mean"].map("{:.1f}".format)
    table["Best", "min"] = table["Best", "min"].astype(int)
    table["Time"] = table["Time"].map(lambda x: round(x, 3))
    table["Time"] = table["Time"].map("{:.3f}".format)
    print(table.to_latex(index=False))

    print()
    print(table)
