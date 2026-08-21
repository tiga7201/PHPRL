from dataclasses import dataclass
from random import Random
from time import perf_counter
from typing import Dict, List, Optional, Tuple

from env.fjspwf_env import ScheduledOp
from env.instance_generator import InstanceData

from .chromosome import Chromosome, random_initialize
from .decoder import HgaDecoder
from .local_search import CriticalPathLocalSearch
from .operators import crossover, mutate


ChromosomeKey = Tuple[
    Tuple[int, ...],
    Tuple[int, ...],
    Tuple[int, ...],
]


@dataclass
class HGAConfig:
    population_size: int = 30
    generations: int = 50
    crossover_rate: float = 0.9

    os_mutation_rate: float = 0.1
    machine_mutation_rate: float = 0.1
    worker_mutation_rate: float = 0.1

    elite_size: int = 2
    tournament_size: int = 3

    # Local search is applied periodically to the best individuals.
    local_search_elite_size: int = 2
    local_search_interval: int = 5

    max_time_seconds: Optional[float] = None
    stagnation_generations: Optional[int] = None
    verbose: bool = True

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError(
                "population_size must be at least 2."
            )

        if self.generations < 1:
            raise ValueError(
                "generations must be at least 1."
            )

        if not 0 <= self.elite_size < self.population_size:
            raise ValueError(
                "elite_size must be between 0 and "
                "population_size - 1."
            )

        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError(
                "tournament_size must be between 2 and "
                "population_size."
            )

        if not (
            0
            <= self.local_search_elite_size
            <= self.population_size
        ):
            raise ValueError(
                "local_search_elite_size must be between 0 and "
                "population_size."
            )

        if self.local_search_interval < 1:
            raise ValueError(
                "local_search_interval must be at least 1."
            )

        for name, rate in (
            ("crossover_rate", self.crossover_rate),
            ("os_mutation_rate", self.os_mutation_rate),
            (
                "machine_mutation_rate",
                self.machine_mutation_rate,
            ),
            (
                "worker_mutation_rate",
                self.worker_mutation_rate,
            ),
        ):
            if not 0.0 <= rate <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

        if (
            self.max_time_seconds is not None
            and self.max_time_seconds <= 0
        ):
            raise ValueError(
                "max_time_seconds must be positive."
            )

        if (
            self.stagnation_generations is not None
            and self.stagnation_generations < 1
        ):
            raise ValueError(
                "stagnation_generations must be at least 1."
            )


@dataclass
class HGAResult:
    best_chromosome: Chromosome
    best_schedule: List[ScheduledOp]
    best_makespan: float
    history: List[Dict[str, float]]
    generations_completed: int

    evaluations: int
    ga_evaluations: int
    local_search_evaluations: int

    elapsed_seconds: float


class HGA:
    def __init__(
        self,
        instance: InstanceData,
        decoder: HgaDecoder,
        config: Optional[HGAConfig] = None,
        local_search: Optional[
            CriticalPathLocalSearch
        ] = None,
        seed: Optional[int] = None,
    ):
        self.instance = instance
        self.decoder = decoder
        self.config = config or HGAConfig()
        self.config.validate()

        self.local_search = local_search
        self.rng = Random(seed)

        # A fixed chromosome has deterministic fitness for the same
        # instance and decoder.
        self._fitness_cache: Dict[
            ChromosomeKey,
            float,
        ] = {}

        self.ga_evaluations = 0
        self.local_search_evaluations = 0

    @staticmethod
    def _chromosome_key(
        chromosome: Chromosome,
    ) -> ChromosomeKey:
        return (
            tuple(chromosome.OS),
            tuple(chromosome.MS),
            tuple(chromosome.WS),
        )

    def _initialize_population(
        self,
    ) -> List[Chromosome]:
        return [
            random_initialize(
                instance=self.instance,
                rng=self.rng,
            )
            for _ in range(self.config.population_size)
        ]

    def _evaluate(
        self,
        chromosome: Chromosome,
    ) -> float:
        if chromosome.fitness is not None:
            return float(chromosome.fitness)

        key = self._chromosome_key(chromosome)

        if key in self._fitness_cache:
            fitness = self._fitness_cache[key]
            chromosome.fitness = fitness
            return fitness

        _, makespan = self.decoder.decode(chromosome)
        fitness = float(makespan)

        chromosome.fitness = fitness
        self._fitness_cache[key] = fitness
        self.ga_evaluations += 1

        return fitness

    def _evaluate_population(
        self,
        population: List[Chromosome],
    ) -> None:
        for chromosome in population:
            self._evaluate(chromosome)

    def _tournament_select(
        self,
        population: List[Chromosome],
    ) -> Chromosome:
        candidates = self.rng.sample(
            population,
            self.config.tournament_size,
        )

        return min(
            candidates,
            key=lambda chromosome: float(
                chromosome.fitness
            ),
        )

    @staticmethod
    def _best(
        population: List[Chromosome],
    ) -> Chromosome:
        return min(
            population,
            key=lambda chromosome: float(
                chromosome.fitness
            ),
        )

    @staticmethod
    def _history_record(
        generation: int,
        population: List[Chromosome],
    ) -> Dict[str, float]:
        fitness_values = [
            float(chromosome.fitness)
            for chromosome in population
        ]

        return {
            "generation": float(generation),
            "best": min(fitness_values),
            "mean": (
                sum(fitness_values)
                / len(fitness_values)
            ),
            "worst": max(fitness_values),
        }

    def _create_next_population(
        self,
        population: List[Chromosome],
    ) -> List[Chromosome]:
        ranked_population = sorted(
            population,
            key=lambda chromosome: float(
                chromosome.fitness
            ),
        )

        next_population = [
            chromosome.clone()
            for chromosome in ranked_population[
                :self.config.elite_size
            ]
        ]

        while (
            len(next_population)
            < self.config.population_size
        ):
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)

            if (
                self.rng.random()
                < self.config.crossover_rate
            ):
                child1, child2 = crossover(
                    parent1=parent1,
                    parent2=parent2,
                    instance=self.instance,
                    rng=self.rng,
                )
            else:
                child1 = parent1.clone()
                child2 = parent2.clone()

            child1 = mutate(
                chromosome=child1,
                instance=self.instance,
                os_rate=(
                    self.config.os_mutation_rate
                ),
                machine_rate=(
                    self.config.machine_mutation_rate
                ),
                worker_rate=(
                    self.config.worker_mutation_rate
                ),
                rng=self.rng,
            )

            child2 = mutate(
                chromosome=child2,
                instance=self.instance,
                os_rate=(
                    self.config.os_mutation_rate
                ),
                machine_rate=(
                    self.config.machine_mutation_rate
                ),
                worker_rate=(
                    self.config.worker_mutation_rate
                ),
                rng=self.rng,
            )

            next_population.append(child1)

            if (
                len(next_population)
                < self.config.population_size
            ):
                next_population.append(child2)

        return next_population

    def _apply_local_search(
        self,
        population: List[Chromosome],
        generation: int,
    ) -> None:
        if self.local_search is None:
            return

        if self.config.local_search_elite_size == 0:
            return

        if (
            generation
            % self.config.local_search_interval
            != 0
        ):
            return

        ranked_indices = sorted(
            range(len(population)),
            key=lambda index: float(
                population[index].fitness
            ),
        )

        selected_indices = ranked_indices[
            :self.config.local_search_elite_size
        ]

        processed_keys = set()

        for population_index in selected_indices:
            chromosome = population[population_index]
            original_key = self._chromosome_key(
                chromosome
            )

            # Avoid applying local search repeatedly to duplicate
            # elite chromosomes in the same generation.
            if original_key in processed_keys:
                continue

            processed_keys.add(original_key)

            result = self.local_search.improve(
                chromosome
            )
            self.local_search_evaluations += (
                result.evaluations
            )

            if not result.improved:
                continue

            improved_chromosome = result.chromosome
            improved_chromosome.fitness = float(
                result.makespan
            )

            population[population_index] = (
                improved_chromosome
            )

            improved_key = self._chromosome_key(
                improved_chromosome
            )
            self._fitness_cache[improved_key] = float(
                result.makespan
            )

            if self.config.verbose:
                print(
                    "  Local search | "
                    f"{chromosome.fitness:.4f} -> "
                    f"{result.makespan:.4f}"
                )

    def _time_limit_reached(
        self,
        start_time: float,
    ) -> bool:
        if self.config.max_time_seconds is None:
            return False

        elapsed = perf_counter() - start_time
        return elapsed >= self.config.max_time_seconds

    def solve(self) -> HGAResult:
        start_time = perf_counter()

        self.ga_evaluations = 0
        self.local_search_evaluations = 0
        self._fitness_cache.clear()

        population = self._initialize_population()
        self._evaluate_population(population)

        global_best = self._best(
            population
        ).clone()

        history = [
            self._history_record(
                generation=0,
                population=population,
            )
        ]

        generations_completed = 0
        stagnant_generations = 0

        if self.config.verbose:
            print(
                "Generation 0 | "
                f"best={global_best.fitness:.4f} | "
                f"mean={history[-1]['mean']:.4f}"
            )

        for generation in range(
            1,
            self.config.generations + 1,
        ):
            if self._time_limit_reached(start_time):
                if self.config.verbose:
                    print(
                        "Stopped because the time limit "
                        "was reached."
                    )
                break

            population = self._create_next_population(
                population
            )
            self._evaluate_population(population)

            self._apply_local_search(
                population=population,
                generation=generation,
            )

            generation_best = self._best(population)
            generations_completed = generation

            if (
                generation_best.fitness
                < global_best.fitness
            ):
                global_best = (
                    generation_best.clone()
                )
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            record = self._history_record(
                generation=generation,
                population=population,
            )
            history.append(record)

            if self.config.verbose:
                print(
                    f"Generation {generation} | "
                    f"best={record['best']:.4f} | "
                    f"mean={record['mean']:.4f} | "
                    f"global={global_best.fitness:.4f}"
                )

            if (
                self.config.stagnation_generations
                is not None
                and stagnant_generations
                >= self.config.stagnation_generations
            ):
                if self.config.verbose:
                    print(
                        "Stopped because no improvement "
                        "was found for "
                        f"{stagnant_generations} "
                        "generations."
                    )
                break

        best_schedule, best_makespan = (
            self.decoder.decode(global_best)
        )
        final_decode_evaluations = 1

        best_makespan = float(best_makespan)
        global_best.fitness = best_makespan

        elapsed_seconds = (
            perf_counter() - start_time
        )

        total_evaluations = (
            self.ga_evaluations
            + self.local_search_evaluations
            + final_decode_evaluations
        )

        return HGAResult(
            best_chromosome=global_best,
            best_schedule=list(best_schedule),
            best_makespan=best_makespan,
            history=history,
            generations_completed=(
                generations_completed
            ),
            evaluations=total_evaluations,
            ga_evaluations=self.ga_evaluations,
            local_search_evaluations=(
                self.local_search_evaluations
            ),
            elapsed_seconds=float(
                elapsed_seconds
            ),
        )