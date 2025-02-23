import numpy as np
from copy import deepcopy
from cvrp_tabu_search.problem import Solution, Instance
from cvrp_tabu_search.utils import prev_vertex, next_vertex, get_route_demand


def update_objective_function_intraswap(w: np.ndarray, old_obj: np.int64, i: int, j: int, rv: list[int]):
    new_obj = old_obj
    v = rv[i]
    u = rv[j]
    v0 = prev_vertex(rv, i)
    v1 = next_vertex(rv, i)
    u0 = prev_vertex(rv, j)
    u1 = next_vertex(rv, j)

    # remove o custo de (v0, v) e (u, u1)
    new_obj -= w[v0, v] + w[u, u1]
    # adiciona o custo de (v0, u) e (v, u1)
    new_obj += w[v0, u] + w[v, u1]
    # se estamos trocando dois índices adjacentes, não precisamos fazer nada
    # mas se são índices não-adjacentes, somamos normalmente
    if i + 1 != j:
        new_obj -= w[v, v1] + w[u0, u]
        new_obj += w[u, v1] + w[u0, v]

    return new_obj


def intraswap_neighborhood(s: Solution, p: Instance):
    # para cada item da rota...
    for i in range(len(s.s)):
        for l, v in enumerate(s.s[i]):
            for m, u in enumerate(s.s[i][l + 1 :], l + 1):
                # cria uma cópia da solução
                new_s = deepcopy(s)

                # troca os itens
                new_s.s[i].insert(l, u)
                new_s.s[i].pop(l + 1)
                new_s.s[i].insert(m, v)
                new_s.s[i].pop(m + 1)

                # atualiza o valor da função objetivo dinamicamente
                new_s.f = update_objective_function_intraswap(p.w, s.f, l, m, s.s[i])
                yield new_s, [(v, i), (u, i)]


def update_objective_function_crossover(w: np.ndarray, old_obj: np.int64, i: int, j: int, rv: list[int], ru: list[int]):
    new_obj = old_obj
    v = rv[i]
    u = ru[j]
    v1 = next_vertex(rv, i)
    u1 = next_vertex(ru, j)
    # remove o custo da conexão entre as partes esquerda e direita de r1
    new_obj -= w[v, v1]
    # adiciona o custo da nova conexão com a direita de r2
    new_obj += w[v, u1]
    # remove o custo da conexão entre as partes esquerda e direita de r2
    new_obj -= w[u, u1]
    # adiciona o custo da nova conexão com a direita de r1
    new_obj += w[u, v1]

    return new_obj


def crossover_neighborhood(s: Solution, p: Instance):
    # para cada combinação r0 x r1, em que r0 != r1
    for i in range(len(s.s)):
        for j in range(len(s.s)):
            if i == j:
                continue

            # para cada item da rota pivô, quebrar a rota no item (ex.: [v0, v] e [v1, v2...])
            for l in range(1, len(s.s[i])):
                # pega a demanda do lado direito de r1
                r1_right_demand = get_route_demand(s.s[i][l:], p.d)

                for m in range(1, len(s.s[j])):
                    # pega a demanda do lado direito de r2
                    r2_right_demand = get_route_demand(s.s[j][m:], p.d)

                    # se os itens podem ser trocados de rota sem estourar a capacidade...
                    new_r1_demand = s.d[i] - r1_right_demand + r2_right_demand
                    new_r2_demand = s.d[j] + r1_right_demand - r2_right_demand

                    if new_r1_demand <= p.c and new_r2_demand <= p.c:
                        # cria uma cópia da solução
                        new_s = deepcopy(s)

                        # remove a parte da direita de r1 e coloca a direita de r2
                        new_r1 = new_s.s[i][:l]
                        new_r1.extend(new_s.s[j][m:])

                        # remove a parte da direita de r2 e coloca a direita de r1
                        new_r2 = new_s.s[j][:m]
                        new_r2.extend(new_s.s[i][l:])

                        # atualiza as rotas
                        new_s.s[i] = new_r1
                        new_s.s[j] = new_r2

                        # atualiza o valor das capacidades dinamicamente
                        new_s.d[i] = new_r1_demand
                        new_s.d[j] = new_r2_demand

                        # atualiza o valor da função objetivo dinamicamente
                        new_s.f = update_objective_function_crossover(p.w, s.f, l - 1, m - 1, s.s[i], s.s[j])
                        yield new_s, [(o, j) for o in s.s[i][l:]] + [(o, i) for o in s.s[j][m:]]


def update_objective_function_swap(w: np.ndarray, old_obj: np.int64, i: int, j: int, rv: list[int], ru: list[int]):
    new_obj = old_obj
    v = rv[i]
    u = ru[j]
    v0 = prev_vertex(rv, i)
    v1 = next_vertex(rv, i)
    u0 = prev_vertex(ru, j)
    u1 = next_vertex(ru, j)
    # remove o custo de (v0, v) e (v, v1)
    new_obj -= w[v0, v] + w[v, v1]
    # adiciona o custo de (v0, u) e (u, v1)
    new_obj += w[v0, u] + w[u, v1]
    # remove o custo de (u0, u) e (u, u1)
    new_obj -= w[u0, u] + w[u, u1]
    # adiciona o custo de (u0, v) e (v, u1)
    new_obj += w[u0, v] + w[v, u1]
    return new_obj


def swap_neighborhood(s: Solution, p: Instance):
    # para cada combinação r0 x r1, em que idx(r0) < idx(r1)
    for i in range(len(s.s)):
        for j in range(i + 1, len(s.s)):
            # para cada item da rota pivô, ver se pode ser inserida em todas as posições de todas as outras rotas
            for l, v in enumerate(s.s[i]):
                v_demand = p.d[v]
                for m, u in enumerate(s.s[j]):
                    u_demand = p.d[u]

                    # se os itens podem ser trocados de rota sem estourar a capacidade...
                    new_i_demand = s.d[i] - v_demand + u_demand
                    new_j_demand = s.d[j] + v_demand - u_demand

                    if new_j_demand <= p.c and new_i_demand <= p.c:
                        # cria uma cópia da solução
                        new_s = deepcopy(s)

                        # remove o item da rota 1 e adiciona o item da rota 2
                        new_s.s[i].insert(l, u)
                        new_s.s[i].remove(v)

                        # remove o item da rota 2 e adiciona o item da rota 1
                        new_s.s[j].insert(m, v)
                        new_s.s[j].remove(u)

                        # atualiza o valor das capacidades dinamicamente
                        new_s.d[i] = new_i_demand
                        new_s.d[j] = new_j_demand

                        # atualiza o valor da função objetivo dinamicamente
                        new_s.f = update_objective_function_swap(p.w, s.f, l, m, s.s[i], s.s[j])
                        yield new_s, [(v, j), (u, i)]


def update_objective_function_shift(w: np.ndarray, old_obj: np.int64, i: int, j: int, rv: list[int], rnv: list[int]):
    new_obj = old_obj
    v = rv[i]
    v0 = prev_vertex(rv, i)
    v1 = next_vertex(rv, i)
    u0 = prev_vertex(rnv, j)
    u1 = next_vertex(rnv, j)
    new_obj -= w[v0, v] + w[v, v1]
    # caso a rota seja vazia
    if v0 != v1:
        new_obj += w[v0, v1]
    new_obj += w[u0, v] + w[v, u1]
    new_obj -= w[u0, u1]
    return new_obj


def shift_neighborhood(s: Solution, p: Instance):
    # para cada combinação r0 x r1, em que r0 != r1
    for i in range(len(s.s)):
        for j in range(len(s.s)):
            if i == j:
                continue

            # para cada item da rota pivô, ver se pode ser inserida em todas as posições de todas as outras rotas
            for l, v in enumerate(s.s[i]):
                v_demand = p.d[v]

                # se o item pode ser inserido na rota j sem estourar a capacidade...
                new_j_demand = s.d[j] + v_demand

                if new_j_demand <= p.c:
                    # para cada lugar possível de inserir o ponto na rota
                    for k in range(len(s.s[j]) + 1):
                        # cria uma cópia da solução
                        new_s = deepcopy(s)

                        # remove o item da rota antiga
                        new_s.s[i].remove(v)

                        # cria uma nova solução com o vértice no ponto k da rota
                        new_s.s[j].insert(k, v)

                        # atualiza o valor das capacidades dinamicamente
                        new_s.d[i] -= v_demand
                        new_s.d[j] = new_j_demand

                        # atualiza o valor da função objetivo dinamicamente
                        new_s.f = update_objective_function_shift(p.w, s.f, l, k, s.s[i], new_s.s[j])
                        yield new_s, [(v, j)]
