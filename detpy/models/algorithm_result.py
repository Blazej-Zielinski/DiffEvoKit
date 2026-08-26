import csv
import datetime
import uuid
from pathlib import Path

from matplotlib import pyplot as plt


class AlgorithmResult:
    def __init__(
            self,
            epoch_metrics,
            avg_fitness,
            std_fitness,
            best_solution
    ):
        self.epoch_metrics = epoch_metrics
        self.avg_fitness = avg_fitness
        self.std_fitness = std_fitness
        self.best_solution = best_solution

    def __repr__(self):
        return (
            f"AlgorithmResult("
            f"avg_fitness={self.avg_fitness}, "
            f"std_fitness={self.std_fitness}, "
            f"best_solution={self.best_solution})"
        )

    @staticmethod
    def _get_csv_path(
            output_dir,
            filename,
            prefix
    ):
        """
        Creates output directory if it does not exist
        and generates a filename if one was not provided.
        """

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        if filename is None:
            timestamp = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            run_id = str(uuid.uuid4())[:6]

            filename = (
                f"{prefix}_"
                f"{timestamp}_"
                f"{run_id}.csv"
            )

        return output_dir / filename

    def save_metrics_to_csv(
            self,
            output_dir="results",
            filename=None
    ):
        """
        Saves metrics from all epochs to CSV.

        Parameters
        ----------
        output_dir : str
            Directory where the CSV file will be saved.

        filename : str or None
            Custom filename.
            If None, filename is generated automatically.

        Returns
        -------
        Path
            Path to the created CSV file.
        """

        if not self.epoch_metrics:
            raise ValueError(
                "No epoch metrics available."
            )

        filepath = self._get_csv_path(
            output_dir=output_dir,
            filename=filename,
            prefix="algorithm_results"
        )

        dimension = len(
            self.epoch_metrics[
                0
            ].best_individual.chromosomes
        )

        headers = [
            "epoch",
            "nfe",
            "best_fitness",
            "worst_fitness",
            "population_mean",
            "population_std",
            "execution_time"
        ]

        # Add x1, x2, ..., xn
        headers.extend(
            f"x{i + 1}"
            for i in range(dimension)
        )

        with open(
                filepath,
                "w",
                newline="",
                encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow(headers)

            for metric in self.epoch_metrics:

                best_values = (
                    metric.best_individual
                    .get_chromosomes()
                )

                writer.writerow([
                    metric.epoch,
                    metric.nfe,
                    metric.best_individual.fitness_value,
                    metric.worst_individual.fitness_value,
                    metric.population_mean,
                    metric.population_std,
                    metric.execution_time,
                    *best_values
                ])

        return filepath

    def save_final_result_to_csv(
            self,
            output_dir="results",
            filename=None
    ):
        """
        Saves only the final optimization result to CSV.

        Parameters
        ----------
        output_dir : str
            Directory where the CSV file will be saved.

        filename : str or None
            Custom filename.
            If None, filename is generated automatically.

        Returns
        -------
        Path
            Path to the created CSV file.
        """

        if self.best_solution is None:
            raise ValueError(
                "Best solution is not available."
            )

        filepath = self._get_csv_path(
            output_dir=output_dir,
            filename=filename,
            prefix="algorithm_final_result"
        )

        best_values = (
            self.best_solution.get_chromosomes()
        )

        headers = [
            "best_fitness",
            "avg_fitness",
            "std_fitness"
        ]

        # Add x1, x2, ..., xn
        headers.extend(
            f"x{i + 1}"
            for i in range(len(best_values))
        )

        with open(
                filepath,
                "w",
                newline="",
                encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow(headers)

            writer.writerow([
                self.best_solution.fitness_value,
                self.avg_fitness,
                self.std_fitness,
                *best_values
            ])

        return filepath

    def plot_results(
            self,
            x_axis,
            best_fitness_values,
            avg_fitness_values,
            std_fitness_values,
            method_name="Method"
    ):
        plt.figure()
        plt.plot(
            x_axis,
            best_fitness_values,
            label="Best Fitness"
        )
        plt.grid(True)

        plt.xlabel(
            "Number function evaluations (NFE)"
        )
        plt.ylabel(
            "Best Fitness Value"
        )
        plt.title(
            f"Best Fitness per NFE - {method_name}"
        )
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(
            x_axis,
            avg_fitness_values,
            label="Average Fitness",
            color="orange"
        )
        plt.grid(True)

        plt.xlabel(
            "Number function evaluations (NFE)"
        )
        plt.ylabel(
            "Average Fitness"
        )
        plt.title(
            f"Average Fitness per NFE - {method_name}"
        )
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(
            x_axis,
            std_fitness_values,
            label="Standard Deviation of Fitness",
            color="green"
        )
        plt.grid(True)

        plt.xlabel(
            "Number function evaluations (NFE)"
        )
        plt.ylabel(
            "Standard Deviation"
        )
        plt.title(
            f"Standard Deviation of Fitness per NFE - {method_name}"
        )
        plt.legend()
        plt.show()