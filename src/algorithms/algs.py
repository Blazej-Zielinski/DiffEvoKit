import numpy as np

from src.algorithms.boudary_fixing import fix_boundary_constraints
from src.algorithms.methods.default_de import mutation, binomial_crossing, selection
from src.algorithms.methods.best_worst import best_worst_mutation, calculate_cr
from src.algorithms.methods.random_locations import rl_mutation
from src.algorithms.methods.novel_modified import nm_mutation, nm_binomial_crossing, nm_selection,\
    nm_calculate_fm_crm, nm_update_f_cr
from src.algorithms.methods.parent_centric import parent_centric_mutation
from src.algorithms.methods.pbx import pbx_mutation, p_best_crossover, calculate_crm, calculate_fm
from src.algorithms.methods.laplace import laplace_mutation
from src.algorithms.methods.tensegrity_structures_de import ts_mutation, ts_selection
from src.algorithms.methods.bidirectional import bi_mutation, bi_binomial_crossing
from src.algorithms.methods.adaptive_params import ad_mutation, ad_binomial_crossing, ad_selection
from src.algorithms.methods.em_de import em_mutation
from src.algorithms.methods.scaling_params import sp_get_f, sp_get_cr, sp_binomial_crossing


def default_alg(pop, config):
    v_pop = mutation(pop, f=config.mutation_factor)

    # boundary constrains
    fix_boundary_constraints(v_pop, config.boundary_constraints_fun)

    u_pop = binomial_crossing(pop, v_pop, cr=config.crossover_rate)

    # Update values before selection
    u_pop.update_fitness_values(lambda params: config.function.eval(params))

    # Select new population
    new_pop, _ = selection(pop, u_pop)

    return new_pop, ()


def best_worst_alg(pop, config, curr_gen):
    """
    Source: https://www.sciencedirect.com/science/article/pii/S0020025512000278
    :param curr_gen:
    :param pop:
    :param config:
    :return:
    """
    # calculate not constant cr depend on generation number
    cr = calculate_cr(curr_gen, config.num_of_epochs)

    v_pop = best_worst_mutation(pop)

    # boundary constrains
    fix_boundary_constraints(v_pop, config.boundary_constraints_fun)

    u_pop = binomial_crossing(pop, v_pop, cr=cr)

    # Update values before selection
    u_pop.update_fitness_values(lambda params: config.function.eval(params))

    # Select new population
    new_pop, _ = selection(pop, u_pop)

    return new_pop, ()


def random_locations_alg(pop, config):
    """
    Source: https://www.sciencedirect.com/science/article/pii/S037722170500281X#aep-section-id9
    :param pop:
    :param config:
    :return:
    """
    v_pop = rl_mutation(pop)

    # boundary constrains
    fix_boundary_constraints(v_pop, config.boundary_constraints_fun)

    u_pop = binomial_crossing(pop, v_pop, cr=config.crossover_rate)

    # Update values before selection
    u_pop.update_fitness_values(lambda params: config.function.eval(params))

    # Select new population
    new_pop, _ = selection(pop, u_pop)

    return new_pop, ()


# to fix but working good
def novel_modified_de(pop, config, novel_data):
    """
    Source: https://www.sciencedirect.com/science/article/pii/S0898122111000460#s000015
    :param pop:
    :param config:
    :param novel_data:
    :return:
    """
    delta_f, delta_cr, sp, flags, f_arr, cr_arr, f_set, cr_set = novel_data

    v_pop = nm_mutation(pop, f_arr)

    # boundary constrains
    fix_boundary_constraints(v_pop, config.boundary_constraints_fun)

    u_pop = nm_binomial_crossing(pop, v_pop, cr_arr)

    # Update values before selection
    u_pop.update_fitness_values(lambda params: config.function.eval(params))

    # Select new population
    new_pop, better_members_indexes = nm_selection(pop, u_pop)

    for i in range(config.population_size):
        if i in better_members_indexes:
            f_set.add(f_arr[i])
            cr_set.add(cr_arr[i])
        else:
            flags[i] += 1

    f_m, cr_m = nm_calculate_fm_crm(f_set, cr_set)

    for i in range(config.population_size):
        if flags[i] == sp:
            if f_set != set() and cr_set != set():
                f_arr[i], cr_arr[i] = nm_update_f_cr(f_m, cr_m, delta_f, delta_cr)
            else:
                f_arr[i] = np.random.uniform(low=0, high=2)
                cr_arr[i] = np.random.uniform(low=0, high=1)
            flags[i] = 0

    return new_pop, (delta_f, delta_cr, sp, flags, f_arr, cr_arr, set(), set())


# not working
def pcx_de(pop, config):
    """
    source: https://ieeexplore.ieee.org/abstract/document/4625261
    :param pop:
    :param config:
    :return:
    """
    v_pop = parent_centric_mutation(pop)

    # boundary constrains
    fix_boundary_constraints(v_pop, config.boundary_constraints_fun)

    u_pop = binomial_crossing(pop, v_pop, cr=config.crossover_rate)

    # Update values before selection
    u_pop.update_fitness_values(lambda params: config.function.eval(params))

    # Select new population
    new_pop = selection(pop, u_pop)

    return new_pop, ()


# poor working
def pbx_de(pop, config, curr_gen, additional_data):
    """
    Source: https://ieeexplore.ieee.org/abstract/document/6046144
    :param curr_gen:
    :param pop:
    :param config:
    :param additional_data:
    :return:
    """
    f_m, cr_m = additional_data

    v_pop, f_success_set = pbx_mutation(pop, config, f_m)

    # boundary constrains
    fix_boundary_constraints(v_pop, config.boundary_constraints_fun)

    u_pop, cr_success_set = p_best_crossover(pop, v_pop, config, cr_m, curr_gen, config.num_of_epochs)

    # Update values before selection
    u_pop.update_fitness_values(lambda params: config.function.eval(params))

    # Select new population
    new_pop = selection(pop, u_pop)

    f_m = calculate_fm(f_m, f_success_set)
    cr_m = calculate_crm(cr_m, cr_success_set)

    return new_pop, (f_m, cr_m)


def laplace_de(pop, config):
    """
    Source: https://www.sciencedirect.com/science/article/pii/S0952197610000424#sec6 , MDE5 scheme
    :param pop:
    :param config:
    :return:
    """
    v_pop = laplace_mutation(pop)

    # boundary constrains
    fix_boundary_constraints(v_pop, config.boundary_constraints_fun)

    u_pop = binomial_crossing(pop, v_pop, cr=config.crossover_rate)

    # Update values before selection
    u_pop.update_fitness_values(lambda params: config.function.eval(params))

    # Select new population
    new_pop, _ = selection(pop, u_pop)

    return new_pop, ()


# poor
def tensegrity_structures_de(pop, config):
    """
    Source: https://www.sciencedirect.com/science/article/pii/S0263822316313320#s0025
    :param pop:
    :param config:
    :return:
    """
    v_pop = ts_mutation(pop)

    # boundary constrains
    fix_boundary_constraints(v_pop, config.boundary_constraints_fun)

    u_pop = binomial_crossing(pop, v_pop, cr=0.8)

    # Update values before selection
    u_pop.update_fitness_values(lambda params: config.function.eval(params))

    # Select new population
    new_pop = ts_selection(pop, u_pop)

    return new_pop, ()


# see potential, but needs optimization
def bidirectional_de(pop, config):
    """
    Source: https://link.springer.com/article/10.1007/s00500-010-0636-5#Sec3
    :param pop:
    :param config:
    :return:
    """
    v_pop, bi_v_pop = bi_mutation(pop, f=config.mutation_factor)

    # boundary constrains
    fix_boundary_constraints(v_pop, config.boundary_constraints_fun)
    fix_boundary_constraints(bi_v_pop, config.boundary_constraints_fun)

    u_pop, bi_u_pop = bi_binomial_crossing(pop, v_pop, bi_v_pop, cr=config.crossover_rate)

    # Update values before selection
    u_pop.update_fitness_values(lambda params: config.function.eval(params))
    bi_u_pop.update_fitness_values(lambda params: config.function.eval(params))

    # Get best members
    better_members, better_count = selection(u_pop, bi_u_pop)

    # Select new population
    new_pop, better_count_2 = selection(pop, better_members)

    return new_pop, ()


# better then default
def adaptive_params_de(pop, config, additional_data):
    """
    Source: https://ieeexplore.ieee.org/abstract/document/4730987
    :param additional_data:
    :param pop:
    :param config:
    :return:
    """

    f_arr, cr_arr, prob_f, prob_cr = additional_data

    v_pop = ad_mutation(pop, f_arr)
    fix_boundary_constraints(v_pop, config.boundary_constraints_fun)
    u_pop = ad_binomial_crossing(pop, v_pop, cr_arr)

    # Update values before selection
    u_pop.update_fitness_values(lambda params: config.function.eval(params))

    # Select new population
    new_pop, f_arr, cr_arr = ad_selection(pop, u_pop, f_arr, cr_arr, prob_f, prob_cr)

    return new_pop, (f_arr, cr_arr, prob_f, prob_cr)


# it is slightly worse then default
def em_de(pop, config):
    """
    Source: https://link.springer.com/article/10.1007/s13042-015-0479-6#Sec8
    :param pop:
    :param config:
    :return:
    """

    v_pop = em_mutation(pop)
    fix_boundary_constraints(v_pop, config.boundary_constraints_fun)
    u_pop = binomial_crossing(pop, v_pop, config.crossover_rate)

    # Update values before selection
    u_pop.update_fitness_values(lambda params: config.function.eval(params))

    # Select new population
    new_pop = selection(pop, u_pop)

    return new_pop


def scaling_params_de(pop, config, curr_gen):
    """
    Source: https://www.scirp.org/journal/paperinformation.aspx?paperid=96749
    :param curr_gen:
    :param pop:
    :param config:
    :return:
    """
    f = sp_get_f(curr_gen, config.num_of_epochs)
    cr_arr = sp_get_cr(pop)

    v_pop = mutation(pop, f)
    fix_boundary_constraints(v_pop, config.boundary_constraints_fun)
    u_pop = sp_binomial_crossing(pop, v_pop, cr_arr)

    # Update values before selection
    u_pop.update_fitness_values(lambda params: config.function.eval(params))

    # Select new population
    new_pop = selection(pop, u_pop)

    return new_pop
