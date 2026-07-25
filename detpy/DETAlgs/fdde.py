from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import FDDEData
from detpy.DETAlgs.methods.methods_fdde import (
    calculate_fitness_ranking,
    calculate_diversity_ranking,
    calculate_final_ranking,
    fdde_mutation,
)
from detpy.DETAlgs.methods.methods_de import crossing, selection
from detpy.models.enums.boundary_constrain import fix_boundary_constraints


class FDDE(BaseAlg):
    """
        FDDE

        Links:
        https://www.sciencedirect.com/science/article/abs/pii/S2210650220304697

        References:
        L. Tang, Y. Dong, J. Liu,
        Differential evolution with an individual-dependent mechanism,
        Swarm and Evolutionary Computation, Volume 61, 2021, 100816
    """

    def __init__(self, params: FDDEData, db_conn=None, db_auto_write=False):
        super().__init__(FDDE.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = params.mutation_factor
        self.crossover_rate = params.crossover_rate
        self.crossing_type = params.crossing_type
        self.max_gen = params.max_nfe // params.population_size

    def next_epoch(self):
        w = min(self._epoch_number / self.max_gen, 1.0)

        fr = calculate_fitness_ranking(self._pop)
        dr = calculate_diversity_ranking(self._pop)
        final_rankings = calculate_final_ranking(fr, dr, w)

        v_pop = fdde_mutation(self._pop, final_rankings, self.mutation_factor)

        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        u_pop = crossing(self._pop, v_pop, cr=self.crossover_rate, crossing_type=self.crossing_type)

        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        new_pop = selection(self._pop, u_pop)

        self._pop = new_pop
