import time
import random
import math
from tqdm import tqdm
from cvrp_tabu_search.problem import Instance, Solution, Run
from cvrp_tabu_search.neighborhoods import shift_neighborhood, intraswap_neighborhood, swap_neighborhood, crossover_neighborhood


def get_best_neighbor(structure_list: list, s: Solution, p: Instance, run: Run, it_invalid: int):
    # guarda o melhor das vizinhanças
    best_solution: Solution = None
    best_solution_movement = None
    best_f: float = math.inf

    # roda todas as estruturas de vizinhança
    for f in structure_list:
        for s_, movement in f(s, p):
            s_: Solution = s_
            invalid_solution = len(s_) > p.k

            invalid_frequency_multiplier = max(1, it_invalid / 100)

            # calcula o bias para soluções inválidas (k maior que o desejado e capacidade acima do permitido)
            invalid_k_bias = (run.invalid_multiplier * s_.min() * invalid_frequency_multiplier) + 1 if invalid_solution else 1
            invalid_k_bias += s_.get_overcapacity(p.c) * run.params.o * invalid_frequency_multiplier

            # confere se é tabu
            if any([i[1] in run.tabu_list[i[0]] for i in movement]):
                # confere se bate o critério de aspiração
                if s_.f < run.best_solution.f and (best_f > s_.f * invalid_k_bias):
                    best_solution = s_
                    best_f = s_.f * invalid_k_bias
                    best_solution_movement = movement

            else:
                # adiciona bias de frequência
                common_bias = sum([run.common_movements[i] for i, _ in movement]) * run.params.f

                # confere se é a melhor solução da vizinhança até agora
                if best_f > s_.f * invalid_k_bias + common_bias:
                    best_solution = s_
                    best_f = s_.f * invalid_k_bias + common_bias
                    best_solution_movement = movement

    return best_solution, best_solution_movement


def run_tabu(p: Instance, max_time: int, run: Run, s: Solution) -> Run:
    t = 0
    it = 1
    pbar = tqdm(total=max_time)

    it_invalid = 1

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
            s_, movement = get_best_neighbor([neighbor_method], s, p, run, it_invalid)
        s = s_

        # atualiza as frequências dos movimentos e a lista tabu
        for i in movement:
            run.common_movements[i[0]] += 1
            run.tabu_list[i[0]].append(i[1])
            run.tabu_tenures[i[0]].append(run.params.t)

        # confere se é inválido e modifica o número de iterações inválidas
        if len(s) > p.k or s.get_overcapacity(p.c) > 0:
            it_invalid += 1
        else:
            it_invalid = 1
            # atualiza melhor global
            if run.best_solution.f > s.f:
                run.best_solution = s

        diff = time.time() - t_s
        t += diff

        pbar.set_description("Iteration %d" % it)
        pbar.update(diff if diff + pbar.n < max_time else max_time - pbar.n)

        run.update_savefile(s_, t)

        it += 1

    run.save()

    return run
