import time
import random
import math
from tqdm import tqdm
from cvrp_tabu_search.problem import Instance, Solution, Run
from cvrp_tabu_search.neighborhoods import shift_neighborhood, intraswap_neighborhood, swap_neighborhood, crossover_neighborhood


def get_best_neighbor(structure_list: list, s: Solution, p: Instance, run: Run):
    # guarda o melhor das vizinhanças
    best_solution: Solution = None
    best_solution_movement = None
    best_f: float = math.inf

    # roda todas as estruturas de vizinhança
    for f in structure_list:
        for s_, movement in f(s, p):
            s_: Solution = s_

            # calcula o bias para soluções com k maior que o permitido
            invalid_k_bias = run.b * s_.min() if len(s_) > p.k else 0

            # calcula o bias para soluções com capacidade maior que a permitida
            invalid_capacity_bias = s_.get_overcapacity(p.c) * run.a

            # confere se é tabu
            if any([i[1] in run.tabu_list[i[0]] for i in movement]):
                # confere se bate o critério de aspiração
                if s_.f < run.best_solution.f and (best_f > s_.f + invalid_k_bias + invalid_capacity_bias):
                    best_solution = s_
                    best_solution_movement = movement
                    best_f = s_.f + invalid_k_bias + invalid_capacity_bias

            else:
                # adiciona bias de frequência
                common_bias = sum([run.common_movements[i] for i, _ in movement]) * run.params.f

                # confere se é a melhor solução da vizinhança até agora
                if best_f > s_.f + invalid_k_bias + common_bias + invalid_capacity_bias:
                    best_solution = s_
                    best_solution_movement = movement
                    best_f = s_.f + invalid_k_bias + common_bias + invalid_capacity_bias

    return best_solution, best_solution_movement


def run_tabu(p: Instance, max_time: int, run: Run, s: Solution, invalid: bool = False) -> Run:
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
        if len(s) > p.k:
            # remove intraswap porque não ajuda a remover a rota extra
            structures = [shift_neighborhood, swap_neighborhood, crossover_neighborhood]
        else:
            structures = [shift_neighborhood, intraswap_neighborhood, swap_neighborhood, crossover_neighborhood]
            # troca para usar os parâmetros de quando a solução é válida
            run.reset_values()

        s_ = None
        while s_ is None:
            # escolhe uma estrutura de vizinhança aleatoriamente
            neighbor_method = random.choice(structures)
            # remove a estrutura para evitar de procurar nela novamente
            structures.remove(neighbor_method)
            # encontra nova solução que respeita o tabu ou o critério de aspiração
            s_, movement = get_best_neighbor([neighbor_method], s, p, run)
        s = s_

        # atualiza as frequências dos movimentos e a lista tabu
        for i in movement:
            run.common_movements[i[0]] += 1
            run.tabu_list[i[0]].append(i[1])
            run.tabu_tenures[i[0]].append(run.params.t)

        over_k = len(s) > p.k
        run.b = max(100, run.b * (1 + run.params.i)) if over_k else min(0.0001, run.b / (1 + run.params.i))

        over_c = s.get_overcapacity(p.c) > 0
        run.a = max(100, run.a * (1 + run.params.i)) if over_c else min(0.0001, run.a / (1 + run.params.i))

        # atualiza melhor global
        if not over_k and not over_c and run.best_solution.f > s.f:
            run.best_solution = s

        diff = time.time() - t_s
        t += diff

        pbar.set_description("Iteration %d" % it)
        pbar.update(diff if diff + pbar.n < max_time else max_time - pbar.n)

        run.update_savefile(s_, t, over_k, over_c)

        if invalid and not over_k:
            break

        it += 1

    run.save()

    return run
