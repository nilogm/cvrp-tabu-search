import os
import json
import typer
import traceback
import random
from itertools import product
from typing_extensions import Annotated
from cvrp_tabu_search.problem import get_instance, Run, Parameters
from cvrp_tabu_search.tabu_search import run_tabu
from cvrp_tabu_search.clarke_wright import clarke_wright
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

    keys = ["instances", "run_time", "valid", "invalid", "seeds"]

    if any([i not in d for i in keys]):
        raise AttributeError(f"Configuration file has missing values. Make sure it contains the following keys: {keys}")

    all_configs = [
        i
        for i in product(
            d["valid"]["tabu_tenure"],
            d["valid"]["frequency_multiplier"],
            d["valid"]["invalid_multiplier"],
            d["invalid"]["tabu_tenure"],
            d["invalid"]["frequency_multiplier"],
            d["invalid"]["invalid_multiplier"],
            d["seeds"],
        )
    ]

    results_folder_path = os.path.join(os.getcwd(), results_folder)
    os.makedirs(results_folder_path, exist_ok=True)

    return d, all_configs, d["invalid_run"] if "invalid_run" in d else False


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


def run(instance_path: str, run_time: int, all_configs: list, results_folder: str, invalid: bool = False):
    """Executa o algoritmo para a instância dada.

    Args:
        instance_path (str): caminho para a instância
        run_time (int): tempo de execução
        all_configs (list): todas as combinações de parâmetros para testar
        results_folder (str): diretório de destino dos resultados
    """
    instance = get_instance(instance_path)
    for v_t, v_f, v_i, i_t, i_f, i_i, seed in all_configs:
        print(instance_path, f" v_t={v_t}; ", f" v_f={v_f}; ", f" v_i={v_i}; ", f" i_t={i_t}; ", f" i_f={i_f}; ", f" i_i={i_i}; ", f" s={seed}; ")

        random.seed(seed)

        # solução inicial
        s = clarke_wright(instance)

        valid_params = Parameters(instance.n, v_t, v_f, v_i)
        invalid_params = Parameters(instance.n, i_t, i_f, i_i)

        run = Run(s, instance.n, valid_params, invalid_params, seed)
        run.begin_savefile(results_folder, instance.name)

        run_tabu(instance, run_time, run, s, invalid)


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
    c, all_configs, invalid = init(config_file, results_folder)

    for instance_path in c["instances"]:
        path = check_instance_path(instance_path)
        if path is None:
            continue

        # se for diretório, executa todos os itens do diretório
        if os.path.isdir(path):
            instances = set([check_instance_path(os.path.join(path, i.removesuffix(".vrp"))) for i in os.listdir(path) if i.endswith(".vrp")])
            for i in sorted(instances):
                path_ = os.path.join(os.getcwd(), i)
                try:
                    run(path_, c["run_time"], all_configs, results_folder, invalid)
                except Exception as e:
                    print(e)
                    print(traceback.format_exc(e))

        # se for um arquivo, executa a instância
        else:
            try:
                run(path, c["run_time"], all_configs, results_folder, invalid)
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

    fig.savefig("graph.png")

    plt.show()


def read_folder(results_folder: Annotated[str, typer.Option(help="Directory containing results .csvs")]):
    folder_path = os.path.join(os.getcwd(), results_folder)
    files = sorted(os.listdir(folder_path))

    cols = ["Instance", "Solution", "Best", "Time", "Iteration", "Tenure", "Frequency", "Invalid", "Invalid Tenure", "Invalid Frequency", "Invalid Invalid", "Gap", "Seed"]
    all_df = pd.DataFrame()

    for i in files:
        instance_name, info = i.split("__")
        instance = load_instance(instance_name)

        df_path = os.path.join(folder_path, i)
        df = pd.read_csv(df_path)

        sol_cost = objective_function(instance.solution["routes"], instance.w)
        sol_cost = instance.solution["cost"]

        _, tenure, _, frequency, _, invalid, _, i_tenure, _, i_frequency, _, i_invalid, _, seed = info.removesuffix(".csv").split("_")

        best = df.iloc[-1]["global"]
        min_iteration = df["local"].idxmin()
        min_time = df.iloc[min_iteration]["time"]
        gap = (best - sol_cost) / sol_cost

        all_df = pd.concat(
            [
                all_df,
                pd.DataFrame(
                    [[instance_name, sol_cost, best, min_time, min_iteration + 1, float(tenure), float(frequency), float(invalid), float(i_tenure), float(i_frequency), float(i_invalid), gap, seed]],
                    columns=cols,
                ),
            ]
        )

    return all_df


@app_experiment.command(help="Shows the parameter tuning tables")
def analyze(results_folder: Annotated[str, typer.Option(help="Directory containing results .csvs")]):
    """Printa a tabela dos resultados dos experimentos com os parâmetros para análise.

    Args:
        results_folder (Annotated[str, typer.Option, optional): caminho para o diretório. Defaults to "Directory containing results .csvs")].
    """
    all_df = read_folder(results_folder)
    all_df = all_df.drop(columns=["Seed"])

    analysis = all_df.sort_values(["Tenure", "Frequency", "Invalid"])
    analysis["Tenure"] = analysis["Tenure"].map("{:.1f}".format)
    analysis["Frequency"] = analysis["Frequency"].map("{:.3f}".format)
    analysis["Invalid"] = analysis["Invalid"].map("{:.1f}".format)
    analysis = analysis.groupby(["Tenure", "Frequency", "Invalid"], sort=False).agg({"Gap": ["min", "mean", "std"]})
    print(analysis.to_latex())
    print()
    print(analysis.sort_values(by=[("Gap", "mean"), ("Gap", "std")]))
    print()

    all_df = all_df.drop(columns=["Solution", "Time", "Iteration", "Gap"])
    all_df = all_df[["Instance", "Tenure", "Frequency", "Invalid", "Best"]]
    table = all_df.sort_values(["Instance", "Tenure", "Frequency", "Invalid"])
    table["Tenure"] = table["Tenure"].map("{:.1f}".format)
    table["Frequency"] = table["Frequency"].map("{:.3f}".format)
    table["Invalid"] = table["Invalid"].map("{:.1f}".format)
    table["Best"] = table["Best"].astype(int)
    print(table.to_latex(index=False))
    print()
    print(table)


@app_experiment.command(help="Shows the invalid parameter tuning tables")
def invalid(results_folder: Annotated[str, typer.Option(help="Directory containing results .csvs")], invalid: bool = False):
    """Printa a tabela dos resultados dos experimentos com os parâmetros para análise.

    Args:
        results_folder (Annotated[str, typer.Option, optional): caminho para o diretório. Defaults to "Directory containing results .csvs")].
    """
    all_df = read_folder(results_folder)
    all_df = all_df.drop(columns=["Seed"])

    analysis = all_df.sort_values(["Invalid Tenure", "Invalid Frequency", "Invalid Invalid"])
    analysis["Invalid Tenure"] = analysis["Invalid Tenure"].map("{:.1f}".format)
    analysis["Invalid Frequency"] = analysis["Invalid Frequency"].map("{:.3f}".format)
    analysis["Invalid Invalid"] = analysis["Invalid Invalid"].map("{:.1f}".format)
    analysis = analysis.groupby(["Invalid Tenure", "Invalid Frequency", "Invalid Invalid"], sort=False).agg({"Iteration": ["min", "mean", "std"]})
    print(analysis.to_latex())
    print()
    print(analysis.sort_values(by=[("Iteration", "mean"), ("Iteration", "std")]))
    print()

    all_df = all_df.drop(columns=["Solution", "Time", "Iteration", "Gap"])
    all_df = all_df[["Instance", "Invalid Tenure", "Invalid Frequency", "Invalid Invalid", "Best"]]
    table = all_df.sort_values(["Instance", "Invalid Tenure", "Invalid Frequency", "Invalid Invalid"])
    table["Invalid Tenure"] = table["Invalid Tenure"].map("{:.1f}".format)
    table["Invalid Frequency"] = table["Invalid Frequency"].map("{:.3f}".format)
    table["Invalid Invalid"] = table["Invalid Invalid"].map("{:.1f}".format)
    table["Best"] = table["Best"].astype(int)
    print(table.to_latex(index=False))
    print()
    print(table)


@app_experiment.command(help="Shows the final results table")
def table(results_folder: Annotated[str, typer.Option(help="Directory containing results .csvs")]):
    """Printa a tabela dos resultados dos experimentos finais.

    Args:
        results_folder (Annotated[str, typer.Option, optional): caminho para o diretório. Defaults to "Directory containing results .csvs")].
    """
    all_df = read_folder(results_folder)
    all_df = all_df.drop(columns=["Tenure", "Frequency"])

    table = all_df.sort_values(["Instance", "Best", "Time"]).groupby(["Instance", "Solution"], as_index=False).agg({"Best": ["first", "mean"], "Time": ["first", "mean"], "Gap": ["min", "mean"]})
    table["Solution"] = table["Solution"].astype(int)
    table["Best", "first"] = table["Best", "first"].astype(int)
    table["Best", "mean"] = table["Best", "mean"].map("{:.1f}".format)
    table["Time"] = table["Time"].map(lambda x: round(x, 3))
    table["Time"] = table["Time"].map("{:.3f}".format)
    print(table.to_latex(index=False))

    print(len(table[table["Gap", "min"] == 0]), len(table[table["Gap", "mean"] == 0]))
    table.to_csv("table.csv")

    print()
    print(table)
