import copy
import random
import numpy as np

from src.models.member import Member
from src.models.population import Population
from src.enums.optimization import OptimizationType


def opp_based_calculate_opposite_pop(population: Population, min_vals: np.ndarray[float] = None,
                                     max_vals: np.ndarray[float] = None) -> Population:
    new_members = []

    if min_vals is not None and max_vals is not None:
        interval = min_vals + max_vals
    else:
        interval = sum(population.interval)

    for member in population.members:
        new_member = Member(population.interval, population.arg_num)
        for i in range(population.arg_num):
            if isinstance(interval, np.ndarray):
                new_member.chromosomes[i].real_value = interval[i] - member.chromosomes[i].real_value
            else:
                new_member.chromosomes[i].real_value = interval - member.chromosomes[i].real_value
        new_members.append(new_member)

    opposite_population = Population(
        interval=population.interval,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    opposite_population.members = np.array(new_members)

    return opposite_population


def opp_based_selection(origin_population: Population, modified_population: Population):
    if origin_population.size != modified_population.size:
        print("Selection: populations have different sizes")
        return None

    if origin_population.optimization != modified_population.optimization:
        print("Selection: populations have different optimization types")
        return None

    optimization = origin_population.optimization
    new_members = []
    for i in range(origin_population.size):
        if optimization == OptimizationType.MINIMIZATION:
            if origin_population.members[i] <= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))

        elif optimization == OptimizationType.MAXIMIZATION:
            if origin_population.members[i] >= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))

    new_population = Population(
        interval=origin_population.interval,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)

    return new_population, ()


def opp_based_min_max_gen(pop: Population) -> tuple[np.ndarray[float], np.ndarray[float]]:
    all_chromosomes = np.array([member.chromosomes for member in pop.members])

    min_values = np.min(all_chromosomes, axis=0)
    max_values = np.max(all_chromosomes, axis=0)

    return min_values, max_values
