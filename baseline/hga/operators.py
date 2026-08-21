from random import Random
from typing import List, Optional, Sequence, Set, Tuple

from env.instance_generator import InstanceData, Operation

from .chromosome import Chromosome, validate_chromosome


def _flatten_operations(instance: InstanceData) -> List[Operation]:
    return [
        operation
        for job_operations in instance.jobs
        for operation in job_operations
    ]


def _fill_pox_child(
    fixed_parent_os: Sequence[int],
    donor_parent_os: Sequence[int],
    selected_jobs: Set[int],
) -> List[int]:
    """
    Preserve selected jobs at their positions in fixed_parent_os and fill
    all remaining positions using the order in donor_parent_os.
    """
    child: List[Optional[int]] = [None] * len(fixed_parent_os)

    for position, job_id in enumerate(fixed_parent_os):
        if job_id in selected_jobs:
            child[position] = job_id

    donor_genes = [
        job_id
        for job_id in donor_parent_os
        if job_id not in selected_jobs
    ]
    donor_index = 0

    for position in range(len(child)):
        if child[position] is None:
            child[position] = donor_genes[donor_index]
            donor_index += 1

    return [int(job_id) for job_id in child]


def pox_crossover(
    parent1: Chromosome,
    parent2: Chromosome,
    instance: InstanceData,
    rng: Optional[Random] = None,
) -> Tuple[List[int], List[int]]:
    """
    Precedence Operation Crossover for the job-based OS representation.

    All occurrences of selected jobs retain their positions from one parent.
    The remaining jobs are inserted according to the order of the other
    parent.
    """
    rng = rng or Random()

    validate_chromosome(parent1, instance)
    validate_chromosome(parent2, instance)

    if instance.num_jobs < 2:
        return parent1.OS.copy(), parent2.OS.copy()

    job_ids = list(range(instance.num_jobs))
    selected_count = rng.randint(1, instance.num_jobs - 1)
    selected_jobs = set(rng.sample(job_ids, selected_count))

    child1_os = _fill_pox_child(
        fixed_parent_os=parent1.OS,
        donor_parent_os=parent2.OS,
        selected_jobs=selected_jobs,
    )
    child2_os = _fill_pox_child(
        fixed_parent_os=parent2.OS,
        donor_parent_os=parent1.OS,
        selected_jobs=selected_jobs,
    )

    return child1_os, child2_os


def resource_uniform_crossover(
    parent1: Chromosome,
    parent2: Chromosome,
    rng: Optional[Random] = None,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Uniform crossover for MS and WS.

    Machine-worker assignments are copied as a pair to avoid unnecessarily
    breaking resource combinations inherited from a parent.
    """
    rng = rng or Random()

    if len(parent1.MS) != len(parent2.MS):
        raise ValueError("Parent MS lengths are different.")

    if len(parent1.WS) != len(parent2.WS):
        raise ValueError("Parent WS lengths are different.")

    child1_ms = []
    child1_ws = []
    child2_ms = []
    child2_ws = []

    for index in range(len(parent1.MS)):
        if rng.random() < 0.5:
            child1_ms.append(parent1.MS[index])
            child1_ws.append(parent1.WS[index])

            child2_ms.append(parent2.MS[index])
            child2_ws.append(parent2.WS[index])
        else:
            child1_ms.append(parent2.MS[index])
            child1_ws.append(parent2.WS[index])

            child2_ms.append(parent1.MS[index])
            child2_ws.append(parent1.WS[index])

    return child1_ms, child1_ws, child2_ms, child2_ws


def crossover(
    parent1: Chromosome,
    parent2: Chromosome,
    instance: InstanceData,
    rng: Optional[Random] = None,
) -> Tuple[Chromosome, Chromosome]:
    """Create two feasible offspring using OS-MS-WS crossover."""
    rng = rng or Random()

    validate_chromosome(parent1, instance)
    validate_chromosome(parent2, instance)

    child1_os, child2_os = pox_crossover(
        parent1,
        parent2,
        instance,
        rng,
    )

    child1_ms, child1_ws, child2_ms, child2_ws = (
        resource_uniform_crossover(parent1, parent2, rng)
    )

    child1 = Chromosome(
        OS=child1_os,
        MS=child1_ms,
        WS=child1_ws,
    )
    child2 = Chromosome(
        OS=child2_os,
        MS=child2_ms,
        WS=child2_ws,
    )

    validate_chromosome(child1, instance)
    validate_chromosome(child2, instance)

    return child1, child2


def mutate_os(
    chromosome: Chromosome,
    rng: Optional[Random] = None,
) -> bool:
    """
    Swap two different job genes.

    Returns True when a mutation was performed.
    """
    rng = rng or Random()

    candidate_pairs = []

    for left in range(len(chromosome.OS)):
        for right in range(left + 1, len(chromosome.OS)):
            if chromosome.OS[left] != chromosome.OS[right]:
                candidate_pairs.append((left, right))

    if not candidate_pairs:
        return False

    left, right = rng.choice(candidate_pairs)
    chromosome.OS[left], chromosome.OS[right] = (
        chromosome.OS[right],
        chromosome.OS[left],
    )
    return True


def mutate_machine(
    chromosome: Chromosome,
    instance: InstanceData,
    rng: Optional[Random] = None,
) -> bool:
    """Assign one operation to a different compatible machine."""
    rng = rng or Random()
    operations = _flatten_operations(instance)

    mutable_indices = [
        index
        for index, operation in enumerate(operations)
        if any(
            machine_id != chromosome.MS[index]
            for machine_id in operation.compatible_machines
        )
    ]

    if not mutable_indices:
        return False

    index = rng.choice(mutable_indices)
    operation = operations[index]

    alternatives = [
        machine_id
        for machine_id in operation.compatible_machines
        if machine_id != chromosome.MS[index]
    ]
    chromosome.MS[index] = rng.choice(alternatives)

    return True


def mutate_worker(
    chromosome: Chromosome,
    instance: InstanceData,
    rng: Optional[Random] = None,
) -> bool:
    """Assign one operation to a different compatible worker."""
    rng = rng or Random()
    operations = _flatten_operations(instance)

    mutable_indices = [
        index
        for index, operation in enumerate(operations)
        if any(
            worker_id != chromosome.WS[index]
            for worker_id in operation.compatible_workers
        )
    ]

    if not mutable_indices:
        return False

    index = rng.choice(mutable_indices)
    operation = operations[index]

    alternatives = [
        worker_id
        for worker_id in operation.compatible_workers
        if worker_id != chromosome.WS[index]
    ]
    chromosome.WS[index] = rng.choice(alternatives)

    return True


def mutate(
    chromosome: Chromosome,
    instance: InstanceData,
    os_rate: float = 0.1,
    machine_rate: float = 0.1,
    worker_rate: float = 0.1,
    rng: Optional[Random] = None,
    inplace: bool = False,
) -> Chromosome:
    """
    Apply OS, machine and worker mutation independently.

    Rates are chromosome-level probabilities. Each triggered mutation changes
    one gene or one pair of OS positions.
    """
    rng = rng or Random()

    for name, rate in (
        ("os_rate", os_rate),
        ("machine_rate", machine_rate),
        ("worker_rate", worker_rate),
    ):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")

    result = chromosome if inplace else chromosome.clone()

    if rng.random() < os_rate:
        mutate_os(result, rng)

    if rng.random() < machine_rate:
        mutate_machine(result, instance, rng)

    if rng.random() < worker_rate:
        mutate_worker(result, instance, rng)

    result.fitness = None
    validate_chromosome(result, instance)

    return result