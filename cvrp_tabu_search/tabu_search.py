import time
import random
from tqdm import tqdm
from cvrp_tabu_search.clarke_wright import clarke_wright
from cvrp_tabu_search.problem import Instance, Solution, Run
from cvrp_tabu_search.neighborhoods import shift_neighborhood, intraswap_neighborhood, swap_neighborhood, crossover_neighborhood


def get_best_neighbor(structure_list: list, s: Solution, p: Instance, run: Run):
    # guarda o melhor das vizinhanças
    best_solution: Solution = None
    best_solution_movement = None

    # roda todas as estruturas de vizinhança
    for f in structure_list:
        for s_, movement in f(s, p):
            s_: Solution = s_
            invalid_solution = len(s_) > p.k

            # calcula o bias para soluções inválidas (k maior que o normal)
            invalid_k_bias = (run.invalid_multiplier * s_.min()) + 1 if invalid_solution else 1

            # confere se é tabu
            if any([i[1] in run.tabu_list[i[0]] for i in movement]):
                # confere se bate o critério de aspiração
                if s_.f < run.best_solution.f and (best_solution is None or best_solution.f > s_.f * invalid_k_bias):
                    best_solution = s_
                    best_solution_movement = movement

            else:
                # adiciona bias de frequência
                common_bias = sum([run.common_movements[i] for i, _ in movement]) * run.bias_multiplier

                # confere se é a melhor solução da vizinhança até agora
                if best_solution is None or best_solution.f > s_.f * invalid_k_bias + common_bias:
                    best_solution = s_
                    best_solution_movement = movement

    return best_solution, best_solution_movement


def run_tabu(p: Instance, max_time: int, tabu_tenure: float, bias_multiplier: float, invalid_multiplier: float, seed: int, results_file: str, invalid_run: bool = False) -> Run:
    # solução inicial
    s = clarke_wright(p)

    run = Run(s, p.n, tabu_tenure, bias_multiplier, invalid_multiplier, seed)
    run.begin_savefile(results_file, p.name)

    random.seed(seed)

    t = 0
    it = 1
    pbar = tqdm(total=max_time)
    while t < max_time:
        t_s = time.time()

        # atualiza tabu tenure
        for k in run.tabu_list.keys():
            i = len(run.tabu_list[k]) - 1
            while i >= 0:
                run.tabu_tenures[k][i] -= 1
                if run.tabu_tenures[k][i] == 0:
                    run.tabu_list[k].pop(i)
                    run.tabu_tenures[k].pop(i)
                i -= 1

        # reune os tipos de estruturas de vizinhança
        structures = [shift_neighborhood, intraswap_neighborhood, swap_neighborhood, crossover_neighborhood]

        s_ = None
        while s_ is None:
            # escolhe uma estrutura de vizinhança aleatoriamente
            neighbor_method = random.choice(structures)
            # remove a estrutura para evitar de procurar nela novamente
            structures.remove(neighbor_method)
            # encontra nova solução que respeita o tabu ou o critério de aspiração
            s_, movement = get_best_neighbor([neighbor_method], s, p, run)
        s = s_
        
        invalid_solution = len(s) > p.k

        # atualiza as frequências dos movimentos e a lista tabu
        for i in movement:
            run.common_movements[i[0]] += 1
            run.tabu_list[i[0]].append(i[1])
            run.tabu_tenures[i[0]].append(run.tabu_tenure_value)

        # atualiza melhor global
        if run.best_solution.f > s.f:
            run.best_solution = s

        diff = time.time() - t_s
        t += diff

        pbar.set_description("Iteration %d" % it)
        pbar.update(diff if diff + pbar.n < max_time else max_time - pbar.n)

        run.update_savefile(s_, t)
        if invalid_run and invalid_solution:
            break
        
        it += 1

    run.save()

    return run
