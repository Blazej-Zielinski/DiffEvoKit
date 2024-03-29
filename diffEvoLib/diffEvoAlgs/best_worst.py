from diffEvoLib.diffEvoAlgs.base import BaseDiffEvoAlg
from diffEvoLib.diffEvoAlgs.data.alg_data import BestWorstData
from diffEvoLib.diffEvoAlgs.methods.methods_best_worst import calculate_cr, best_worst_mutation
from diffEvoLib.diffEvoAlgs.methods.methods_default import binomial_crossing, selection
from diffEvoLib.models.enums.boundary_constrain import fix_boundary_constraints


class BestWorst(BaseDiffEvoAlg):
    """
    Source: https://www.sciencedirect.com/science/article/pii/S0020025512000278
    """

    def __init__(self, params: BestWorstData, db_conn=None, db_auto_write=False):
        super().__init__(BestWorst.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr

    def next_epoch(self):
        # Calculate not constant cr depend on generation number
        cr = calculate_cr(self._epoch_number, self.num_of_epochs)

        # New population after mutation
        v_pop = best_worst_mutation(self._pop)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = binomial_crossing(self._pop, v_pop, cr=cr)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
