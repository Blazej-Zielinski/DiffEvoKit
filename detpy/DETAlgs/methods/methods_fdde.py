import random
import copy

import numpy as np

from detpy.models.enums.optimization import OptimizationType
from detpy.models.population import Population


def calculate_fitness_ranking(population: Population) -> np.ndarray:
    """
    FR_i = i, i = 1, 2, ..., NP
    Sort individuals by fitness (best first); rank 1 = best.
    For minimization: ascending order. For maximization: descending order.
    """
    fitness_values = np.array([m.fitness_value for m in population.members])

    if population.optimization == OptimizationType.MINIMIZATION:
        order = np.argsort(fitness_values)
    else:
        order = np.argsort(-fitness_values)

    fr = np.empty(population.size, dtype=float)
    fr[order] = np.arange(1, population.size + 1, dtype=float)
    return fr


def calculate_diversity_ranking(population: Population) -> np.ndarray:
    """
    f_de,i = |f_i - f_mid|
    DR_i = NP - i, i = 1, 2, ..., NP
    Sort individuals by f_de in ascending order; DR = NP - rank_position.
    """
    fitness_values = np.array([m.fitness_value for m in population.members])
    f_mid = np.median(fitness_values)
    f_de = np.abs(fitness_values - f_mid)

    order = np.argsort(f_de)
    dr = np.empty(population.size, dtype=float)
    dr[order] = np.arange(population.size - 1, -1, -1, dtype=float)
    return dr


def calculate_final_ranking(fr: np.ndarray, dr: np.ndarray, w: float) -> np.ndarray:
    """
    R_i = w * DR_i + (1 - w) * FR_i
    w = G / Maxgen
    """
    return w * dr + (1.0 - w) * fr


def fdde_mutation(population: Population, final_rankings: np.ndarray, f: float) -> Population:
    """
    V_i^G = X_r1*^G + F * (X_r2*^G - X_r3*^G)
    r1, r2, r3 randomly selected (distinct, != i), then sorted by final ranking ascending.
    X_r1* = best ranked (lowest R), X_r3* = worst ranked (highest R).
    """
    new_members = []

    for i in range(population.size):
        candidates = list(range(population.size))
        candidates.remove(i)
        r1, r2, r3 = random.sample(candidates, 3)

        triplet = [(r1, final_rankings[r1]), (r2, final_rankings[r2]), (r3, final_rankings[r3])]
        triplet.sort(key=lambda x: x[1])

        idx_best = triplet[0][0]
        idx_mid = triplet[1][0]
        idx_worst = triplet[2][0]

        base_member = population.members[idx_best]
        mid_member = population.members[idx_mid]
        worst_member = population.members[idx_worst]

        new_member = copy.deepcopy(base_member)
        new_member.chromosomes = (
            base_member.chromosomes
            + f * (mid_member.chromosomes - worst_member.chromosomes)
        )
        new_members.append(new_member)

    new_population = Population(
        lb=population.lb,
        ub=population.ub,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population
