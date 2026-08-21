from dataclasses import dataclass
from random import Random
from typing import List, Optional, Sequence, Set, Tuple

from env.fjspwf_env import ScheduledOp
from env.instance_generator import InstanceData

from .chromosome import (
    Chromosome,
    global_operation_index,
    validate_chromosome,
)
from .decoder import HgaDecoder


OperationKey = Tuple[int, int]


@dataclass
class LocalSearchConfig:
    max_iterations: int = 10
    max_neighbors: int = 30
    strategy: str = "best"
    improve_tolerance: float = 1e-9

    use_os_neighbors: bool = True
    use_machine_neighbors: bool = True
    use_worker_neighbors: bool = True

    def validate(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")

        if self.max_neighbors < 1:
            raise ValueError("max_neighbors must be at least 1.")

        if self.strategy not in {"first", "best"}:
            raise ValueError(
                "strategy must be either 'first' or 'best'."
            )

        if self.improve_tolerance < 0:
            raise ValueError("improve_tolerance cannot be negative.")


@dataclass
class LocalSearchResult:
    chromosome: Chromosome
    schedule: List[ScheduledOp]
    makespan: float
    iterations: int
    evaluations: int
    improved: bool


def _operation_key(operation: ScheduledOp) -> OperationKey:
    return operation.job_id, operation.op_id


def extract_critical_path(
    schedule: Sequence[ScheduledOp],
    tolerance: float = 1e-7,
) -> List[ScheduledOp]:
    """
    Extract one critical path using job, machine and worker predecessors.

    A predecessor is considered critical when its completion time equals the
    start time of the current operation.
    """
    if not schedule:
        return []

    operation_by_key = {
        _operation_key(operation): operation
        for operation in schedule
    }

    machine_operations = {}
    worker_operations = {}

    for operation in schedule:
        machine_operations.setdefault(
            operation.machine_id,
            [],
        ).append(operation)
        worker_operations.setdefault(
            operation.worker_id,
            [],
        ).append(operation)

    for operations in machine_operations.values():
        operations.sort(key=lambda op: (op.start, op.end))

    for operations in worker_operations.values():
        operations.sort(key=lambda op: (op.start, op.end))

    current = max(schedule, key=lambda operation: operation.end)
    reverse_path = [current]
    visited = {_operation_key(current)}

    while True:
        predecessor_candidates = []

        if current.op_id > 0:
            job_predecessor = operation_by_key.get(
                (current.job_id, current.op_id - 1)
            )
            if job_predecessor is not None:
                predecessor_candidates.append(job_predecessor)

        for operation in machine_operations[current.machine_id]:
            if operation is current:
                continue

            if operation.end <= current.start + tolerance:
                predecessor_candidates.append(operation)

        for operation in worker_operations[current.worker_id]:
            if operation is current:
                continue

            if operation.end <= current.start + tolerance:
                predecessor_candidates.append(operation)

        tight_predecessors = [
            operation
            for operation in predecessor_candidates
            if abs(operation.end - current.start) <= tolerance
            and _operation_key(operation) not in visited
        ]

        if not tight_predecessors:
            break

        current = max(
            tight_predecessors,
            key=lambda operation: operation.end,
        )
        reverse_path.append(current)
        visited.add(_operation_key(current))

    return list(reversed(reverse_path))


def _find_os_position(
    chromosome: Chromosome,
    job_id: int,
    op_id: int,
) -> int:
    occurrence = 0

    for position, os_job_id in enumerate(chromosome.OS):
        if os_job_id != job_id:
            continue

        if occurrence == op_id:
            return position

        occurrence += 1

    raise ValueError(
        f"Operation ({job_id}, {op_id}) is missing from OS."
    )


def _chromosome_key(
    chromosome: Chromosome,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    return (
        tuple(chromosome.OS),
        tuple(chromosome.MS),
        tuple(chromosome.WS),
    )


def generate_critical_neighbors(
    chromosome: Chromosome,
    critical_path: Sequence[ScheduledOp],
    instance: InstanceData,
    config: LocalSearchConfig,
) -> List[Chromosome]:
    neighbors = []
    seen: Set[
        Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]
    ] = set()

    original_key = _chromosome_key(chromosome)
    seen.add(original_key)

    def add_neighbor(neighbor: Chromosome) -> None:
        neighbor.fitness = None
        key = _chromosome_key(neighbor)

        if key in seen:
            return

        validate_chromosome(neighbor, instance)
        seen.add(key)
        neighbors.append(neighbor)

    if config.use_os_neighbors:
        for left_operation, right_operation in zip(
            critical_path,
            critical_path[1:],
        ):
            if left_operation.job_id == right_operation.job_id:
                continue

            left_position = _find_os_position(
                chromosome,
                left_operation.job_id,
                left_operation.op_id,
            )
            right_position = _find_os_position(
                chromosome,
                right_operation.job_id,
                right_operation.op_id,
            )

            neighbor = chromosome.clone()
            neighbor.OS[left_position], neighbor.OS[right_position] = (
                neighbor.OS[right_position],
                neighbor.OS[left_position],
            )
            add_neighbor(neighbor)

    for scheduled_operation in critical_path:
        job_id = scheduled_operation.job_id
        op_id = scheduled_operation.op_id
        operation = instance.jobs[job_id][op_id]

        gene_index = global_operation_index(
            instance,
            job_id,
            op_id,
        )

        if config.use_machine_neighbors:
            for machine_id in operation.compatible_machines:
                if machine_id == chromosome.MS[gene_index]:
                    continue

                neighbor = chromosome.clone()
                neighbor.MS[gene_index] = machine_id
                add_neighbor(neighbor)

        if config.use_worker_neighbors:
            for worker_id in operation.compatible_workers:
                if worker_id == chromosome.WS[gene_index]:
                    continue

                neighbor = chromosome.clone()
                neighbor.WS[gene_index] = worker_id
                add_neighbor(neighbor)

    return neighbors


class CriticalPathLocalSearch:
    def __init__(
        self,
        instance: InstanceData,
        decoder: HgaDecoder,
        config: Optional[LocalSearchConfig] = None,
        seed: Optional[int] = None,
    ):
        self.instance = instance
        self.decoder = decoder
        self.config = config or LocalSearchConfig()
        self.config.validate()
        self.rng = Random(seed)

    def improve(
        self,
        chromosome: Chromosome,
    ) -> LocalSearchResult:
        validate_chromosome(chromosome, self.instance)

        best = chromosome.clone()
        best_schedule, best_makespan = self.decoder.decode(best)
        best.fitness = float(best_makespan)

        initial_makespan = float(best_makespan)
        evaluations = 1
        completed_iterations = 0

        for iteration in range(1, self.config.max_iterations + 1):
            critical_path = extract_critical_path(best_schedule)

            if not critical_path:
                break

            neighbors = generate_critical_neighbors(
                chromosome=best,
                critical_path=critical_path,
                instance=self.instance,
                config=self.config,
            )

            if not neighbors:
                break

            self.rng.shuffle(neighbors)
            neighbors = neighbors[:self.config.max_neighbors]

            selected_chromosome = None
            selected_schedule = None
            selected_makespan = float(best_makespan)

            for neighbor in neighbors:
                schedule, makespan = self.decoder.decode(neighbor)
                makespan = float(makespan)
                evaluations += 1

                if (
                    makespan
                    < selected_makespan
                    - self.config.improve_tolerance
                ):
                    selected_chromosome = neighbor
                    selected_schedule = schedule
                    selected_makespan = makespan

                    if self.config.strategy == "first":
                        break

            if selected_chromosome is None:
                break

            best = selected_chromosome
            best_schedule = selected_schedule
            best_makespan = selected_makespan
            best.fitness = best_makespan
            completed_iterations = iteration

        return LocalSearchResult(
            chromosome=best,
            schedule=list(best_schedule),
            makespan=float(best_makespan),
            iterations=completed_iterations,
            evaluations=evaluations,
            improved=(
                best_makespan
                < initial_makespan
                - self.config.improve_tolerance
            ),
        )