import json
import typing

from detpy.helpers.metric_helper import Metric
from detpy.models.member import Member


def get_table_name(func_name, alg_name, nr_of_args, pop_size, timestamp, run_id):
    table_name = f"{func_name}_{alg_name}_" \
                 f"args{nr_of_args}_pop{pop_size}_" \
                 f"{timestamp}_{run_id}_results"

    return table_name


def format_individuals(individuals: typing.List[Metric]):
    formatted_individuals = []
    for data in individuals:
        epoch = int(data.epoch) if data.epoch is not None else 0
        nfe = int(data.nfe) if data.nfe is not None else 0
        best_member: Member = data.best_individual
        worst_member: Member = data.worst_individual
        mean = float(data.population_mean) if data.population_mean is not None else 0.0
        std = float(data.population_std) if data.population_std is not None else 0.0
        exec_time = float(data.execution_time) if data.execution_time is not None else 0.0

        population = []
        if data.population is not None:
            population = [str(member) for member in data.population]

        formatted_individuals.append(
            (
                epoch,
                nfe,
                json.dumps([float(c.real_value) for c in best_member.chromosomes]),
                float(best_member.fitness_value),
                json.dumps([float(c.real_value) for c in worst_member.chromosomes]),
                float(worst_member.fitness_value),
                mean,
                std,
                exec_time,
                json.dumps(population)
            )
        )
    return formatted_individuals
