import copy
from typing import List

import numpy as np

from detpy.DETAlgs.archive_reduction.archive_reduction import ArchiveReduction
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.crossover_methods.binomial_crossover import BinomialCrossover
from detpy.DETAlgs.data.alg_data import ILShadeData
from detpy.DETAlgs.math.math_functions import MathFunctions
from detpy.DETAlgs.mutation_methods.current_to_pbest_1 import MutationCurrentToPBest1
from detpy.DETAlgs.random.index_generator import IndexGenerator
from detpy.DETAlgs.random.random_value_generator import RandomValueGenerator
from detpy.models.enums.boundary_constrain import fix_boundary_constraints_with_parent
from detpy.models.enums.ilshade_p_update import ILShadePUpdateStrategy
from detpy.models.enums.optimization import OptimizationType

from detpy.models.population import Population


class ILSHADE(BaseAlg):
    """
        iL-SHADE: Improved L-SHADE Algorithm for Single Objective Real-Parameter Optimization

        Links:
        https://ieeexplore.ieee.org/document/7743922

        References:
        Brest, J., Sepesy Maučec, M., & Boškovič, B. (2016). iL-SHADE: Improved L-SHADE algorithm for single objective
        real-parameter optimization. In 2016 IEEE Congress on Evolutionary Computation (CEC) (pp. 1188–1195). IEEE.
    """

    _FIXED_MEMORY_F = 0.9
    _FIXED_MEMORY_CR = 0.9
    _INITIAL_MEMORY_F = 0.5
    _INITIAL_MEMORY_CR = 0.8
    _TERMINAL = -1.0  # Terminal value for M_CR (Algorithm 4); reset to 0.0 in update_memory

    def __init__(self, params: ILShadeData, db_conn=None, db_auto_write=False):
        super().__init__(ILSHADE.__name__, params, db_conn, db_auto_write)

        self._H = params.memory_size  # Memory size for F and Cr adaptation
        self._memory_F = np.full(self._H, self._INITIAL_MEMORY_F)  # Initial memory for F
        self._memory_Cr = np.full(self._H, self._INITIAL_MEMORY_CR)  # Initial memory for Cr (0.8 in iL-SHADE)
        self._fixed_memory_index = self._H - 1  # Last memory slot with fixed F/Cr values
        self._memory_F[self._fixed_memory_index] = self._FIXED_MEMORY_F
        self._memory_Cr[self._fixed_memory_index] = self._FIXED_MEMORY_CR

        self._p_max = params.p_max
        self._p_min = params.p_min
        self._p_update_strategy = params.p_update_strategy
        self._p = self._p_max  # Current p-best fraction for current-to-pBest/1
        self._k_index = 0

        self._successCr = []
        self._successF = []
        self._difference_fitness_success = []

        self._archive_size = self.population_size  # Size of the archive
        self._archive = []  # Archive for storing replaced members

        self._min_pop_size = params.minimum_population_size  # Minimal population size
        self._start_population_size = self.population_size
        self._g_max = max(1, int(np.ceil(self.nfe_max / self._start_population_size)))  # Estimated max generations
        self._population_size_reduction_strategy = params.population_reduction_strategy

        self._EPSILON = 0.00001  # Tolerance for checking close to zero in update_memory

        self._index_gen = IndexGenerator()
        self._random_value_gen = RandomValueGenerator()
        self._binomial_crossing = BinomialCrossover()
        self._archive_reduction = ArchiveReduction()

    def _update_p(self):
        """
        Update the p-best fraction for current-to-pBest/1 mutation.

        The update strategy is controlled by ``ILShadeData.p_update_strategy``:
        - DECREASING: linear decrease from p_max to p_min (paper Section IV).
        - INCREASING: linear increase from p_min to p_max (paper Equation 1).
        """
        if self._p_update_strategy == ILShadePUpdateStrategy.DECREASING:
            self._p = self._p_max - ((self._p_max - self._p_min) / self.nfe_max) * self.nfe
        else:
            self._p = ((self._p_max - self._p_min) / self.nfe_max) * self.nfe + self._p_min

    def _progress_fraction(self) -> float:
        """
        Return the current generation progress as g / G_max.

        Used for early-stage F and CR constraints.

        Returns:
        - float: Fraction of estimated maximum generations completed.
        """
        return self._epoch_number / self._g_max

    def update_population_size(self, nfe: int, total_nfe: int, start_pop_size: int, min_pop_size: int):
        """
        Calculate new population size using Linear Population Size Reduction (LPSR).

        Parameters:
        - nfe (int): The current number of function evaluations.
        - total_nfe (int): The total number of function evaluations.
        - start_pop_size (int): The initial population size.
        - min_pop_size (int): The minimum population size.
        """
        new_size = self._population_size_reduction_strategy.get_new_population_size(
            nfe, total_nfe, start_pop_size, min_pop_size
        )
        self._pop.resize(new_size)

    def mutate(self,
               population: Population,
               the_best_to_select_table: List[int],
               f_table: List[float]
               ) -> Population:
        """
        Perform mutation step for the population using current-to-pBest/1.

        Parameters:
        - population (Population): The population to mutate.
        - the_best_to_select_table (List[int]): Number of top members to consider as p-best for each individual.
        - f_table (List[float]): List of scaling factors for mutation.

        Returns:
        - Population: A new population with mutated members.
        """
        new_members = []
        sum_archive_and_population = np.concatenate((population.members, self._archive))

        for i in range(population.size):
            r1 = self._index_gen.generate_unique(len(population.members), [i])
            r2 = self._index_gen.generate_unique(len(sum_archive_and_population), [i, r1])

            best_members = population.get_best_members(the_best_to_select_table[i])
            random_index = self._index_gen.generate(0, len(best_members))
            selected_best_member = best_members[random_index]

            mutated_member = MutationCurrentToPBest1.mutate(
                base_member=population.members[i],
                best_member=selected_best_member,
                r1=population.members[r1],
                r2=sum_archive_and_population[r2],
                f=f_table[i]
            )

            new_members.append(mutated_member)

        return Population.with_new_members(population, new_members)

    def _selection(self, origin_population, modified_population, ftable, cr_table):
        """
        Perform the selection step for the iL-SHADE algorithm.

        Selects members for the next generation based on fitness values.
        If the trial vector is better than the target vector, it replaces the target
        and the replaced vector is stored in the archive together with successful F and Cr values.

        Parameters:
        - origin_population (Population): The original population before selection.
        - modified_population (Population): The trial population after mutation and crossover.
        - ftable (List[float]): Scaling factors used during mutation.
        - cr_table (List[float]): Crossover rates used during crossover.

        Returns:
        - Population: A new population containing the selected members for the next generation.
        """
        optimization = origin_population.optimization
        new_members = []

        if optimization == OptimizationType.MINIMIZATION:
            is_better = lambda orig, mod: mod.fitness_value < orig.fitness_value
            diff = lambda orig, mod: orig.fitness_value - mod.fitness_value
        else:
            is_better = lambda orig, mod: mod.fitness_value > orig.fitness_value
            diff = lambda orig, mod: mod.fitness_value - orig.fitness_value

        for i in range(origin_population.size):
            orig = origin_population.members[i]
            mod = modified_population.members[i]

            if not is_better(orig, mod):
                new_members.append(copy.deepcopy(orig))
                continue

            self._archive.append(copy.deepcopy(orig))
            self._successF.append(ftable[i])
            self._successCr.append(cr_table[i])
            self._difference_fitness_success.append(diff(orig, mod))
            new_members.append(copy.deepcopy(mod))

        return Population.with_new_members(origin_population, new_members)

    def update_memory(self, success_f: List[float], success_cr: List[float], difference_fitness_success: List[float]):
        """
        Update historical memory for F and Cr based on successful trial vectors.

        iL-SHADE uses the average of the weighted Lehmer mean and the previous memory value.
        Terminal Cr values are reset to 0.0 instead of being kept as a special marker.
        The fixed memory slot at index H is never updated.

        Parameters:
        - success_f (List[float]): Scaling factors that led to better trial vectors.
        - success_cr (List[float]): Crossover rates that led to better trial vectors.
        - difference_fitness_success (List[float]): Absolute fitness improvements of successful trials.
        """
        if len(success_f) == 0 or len(success_cr) == 0:
            return

        if self._k_index == self._fixed_memory_index:
            self._successF = []
            self._successCr = []
            self._difference_fitness_success = []
            self._k_index = (self._k_index + 1) % self._H
            return

        total = np.sum(difference_fitness_success)
        weights = difference_fitness_success / total
        old_cr = self._memory_Cr[self._k_index]
        old_f = self._memory_F[self._k_index]

        if old_cr < 0 or np.max(success_cr) == 0 or np.isclose(total, 0.0, atol=self._EPSILON):
            self._memory_Cr[self._k_index] = 0.0
        else:
            cr_new = MathFunctions.calculate_lehmer_mean(np.array(success_cr), weights, p=2)
            cr_new = np.clip((cr_new + old_cr) / 2, 0, 1)
            self._memory_Cr[self._k_index] = cr_new

        f_new = MathFunctions.calculate_lehmer_mean(np.array(success_f), weights, p=2)
        f_new = np.clip((f_new + old_f) / 2, 0, 1)
        self._memory_F[self._k_index] = f_new

        self._successF = []
        self._successCr = []
        self._difference_fitness_success = []
        self._k_index = (self._k_index + 1) % self._H

    def _get_memory_values(self, ri: int) -> tuple[float, float]:
        """
        Return the F and Cr memory values for a selected memory index.

        When the fixed memory slot H is selected, returns constant values 0.9/0.9
        regardless of stored memory contents (Algorithm 3, lines 10-13).

        Parameters:
        - ri (int): Selected memory index in range [0, H-1].

        Returns:
        - tuple[float, float]: Mean F and mean Cr for parameter generation.
        """
        if ri == self._fixed_memory_index:
            return self._FIXED_MEMORY_F, self._FIXED_MEMORY_CR
        return self._memory_F[ri], self._memory_Cr[ri]

    def _apply_early_stage_constraints(self, f: float, cr: float) -> tuple[float, float]:
        """
        Apply early-stage constraints to generated F and Cr values.

        Limits high F and low Cr during the first 75% of estimated generations.

        Parameters:
        - f (float): Generated scaling factor.
        - cr (float): Generated crossover rate.

        Returns:
        - tuple[float, float]: Constrained F and Cr values.
        """
        progress = self._progress_fraction()

        if progress < 0.25:
            cr = max(cr, 0.5)
            f = min(f, 0.7)
        elif progress < 0.5:
            cr = max(cr, 0.25)
            f = min(f, 0.8)
        elif progress < 0.75:
            f = min(f, 0.9)

        return f, cr

    def initialize_parameters_for_epoch(self):
        """
        Initialize F, Cr, and p-best parameters for the next epoch .

        For each individual, a random memory index is selected and used to generate
        new F and Cr values. Early-stage constraints are applied afterwards.

        Returns:
        - f_table (List[float]): Scaling factors for mutation.
        - cr_table (List[float]): Crossover rates.
        - the_bests_to_select (List[int]): Number of p-best members to select for each individual.
        """
        f_table = []
        cr_table = []
        the_bests_to_select = []

        for _ in range(self._pop.size):
            ri = np.random.randint(0, self._H)
            mean_f, mean_cr = self._get_memory_values(ri)

            if mean_cr < 0:
                cr = 0.0
            else:
                cr = self._random_value_gen.generate_normal(mean_cr, 0.1, 0.0, 1.0)

            f = self._random_value_gen.generate_cauchy_greater_than_zero(mean_f, 0.1, 0.0, 1.0)
            f, cr = self._apply_early_stage_constraints(f, cr)

            f_table.append(f)
            cr_table.append(cr)

            the_bests_to_select.append(int(self.population_size * self._p))

        return f_table, cr_table, the_bests_to_select

    def next_epoch(self):
        """
        Perform the next epoch of the iL-SHADE algorithm.

        Executes parameter initialization, mutation, crossover, selection, archive reduction,
        memory update, population size reduction, and dynamic p update.
        """
        self._successF = []
        self._successCr = []
        self._difference_fitness_success = []

        f_table, cr_table, the_bests_to_select = self.initialize_parameters_for_epoch()

        mutant = self.mutate(self._pop, the_bests_to_select, f_table)
        trial = self._binomial_crossing.crossover_population(self._pop, mutant, cr_table)

        fix_boundary_constraints_with_parent(self._pop, trial, self.boundary_constraints_fun)
        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        new_pop = self._selection(self._pop, trial, f_table, cr_table)

        self._archive_size = self.population_size
        self._archive = self._archive_reduction.reduce_archive(self._archive, self._archive_size, self.population_size)

        self._pop = new_pop
        self.update_memory(self._successF, self._successCr, self._difference_fitness_success)

        self.update_population_size(self.nfe, self.nfe_max, self._start_population_size, self._min_pop_size)
        self._update_p()
