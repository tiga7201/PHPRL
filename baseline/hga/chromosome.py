from dataclasses import dataclass
from random import Random
from typing import List, Optional, Tuple

from env.instance_generator import InstanceData, Operation


@dataclass
class Chromosome:
    """
    OS-MS-WS chromosome.

    OS:
        Job IDs in scheduling order. The nth occurrence of job j represents
        the nth operation of that job.

    MS / WS:
        Machine and worker assignments indexed by the global operation index.
        Therefore, changing OS does not change the assignment belonging to
        each operation.
    """

    OS: List[int]
    MS: List[int]
    WS: List[int]
    fitness: Optional[float] = None

    def clone(self) -> "Chromosome":
        return Chromosome(
            OS=self.OS.copy(),
            MS=self.MS.copy(),
            WS=self.WS.copy(),
            fitness=self.fitness,
        )


def operation_offsets(instance: InstanceData) -> List[int]:
    """Return the starting global operation index of every job."""
    offsets = []
    offset = 0

    for job_ops in instance.jobs:
        offsets.append(offset)
        offset += len(job_ops)

    return offsets


def total_operations(instance: InstanceData) -> int:
    return sum(len(job_ops) for job_ops in instance.jobs)


def global_operation_index(
    instance: InstanceData,
    job_id: int,
    op_id: int,
) -> int:
    offsets = operation_offsets(instance)
    return offsets[job_id] + op_id


def operation_from_os_position(
    chromosome: Chromosome,
    instance: InstanceData,
    position: int,
) -> Tuple[Operation, int]:
    """
    Resolve an OS position to its operation and global operation index.
    """
    job_id = chromosome.OS[position]
    op_id = chromosome.OS[:position].count(job_id)

    if job_id < 0 or job_id >= instance.num_jobs:
        raise ValueError(f"Invalid job ID in OS: {job_id}")

    if op_id >= len(instance.jobs[job_id]):
        raise ValueError(
            f"Too many occurrences of job {job_id} in OS."
        )

    operation = instance.jobs[job_id][op_id]
    index = global_operation_index(instance, job_id, op_id)

    return operation, index


def random_initialize(
    instance: InstanceData,
    rng: Optional[Random] = None,
) -> Chromosome:
    """
    Generate a legal random OS-MS-WS chromosome.

    Job precedence is represented implicitly: occurrences of each job in OS
    are decoded as operation 0, operation 1, and so on.
    """
    rng = rng or Random()

    os_gene = []
    ms_gene = []
    ws_gene = []

    for job_id, job_ops in enumerate(instance.jobs):
        os_gene.extend([job_id] * len(job_ops))

        for operation in job_ops:
            if not operation.compatible_machines:
                raise ValueError(
                    f"Operation ({job_id}, {operation.op_id}) "
                    "has no compatible machine."
                )

            if not operation.compatible_workers:
                raise ValueError(
                    f"Operation ({job_id}, {operation.op_id}) "
                    "has no compatible worker."
                )

            ms_gene.append(rng.choice(operation.compatible_machines))
            ws_gene.append(rng.choice(operation.compatible_workers))

    rng.shuffle(os_gene)

    chromosome = Chromosome(
        OS=os_gene,
        MS=ms_gene,
        WS=ws_gene,
    )
    validate_chromosome(chromosome, instance)

    return chromosome


def validate_chromosome(
    chromosome: Chromosome,
    instance: InstanceData,
) -> None:
    """Raise ValueError if the chromosome is structurally infeasible."""
    expected_operations = total_operations(instance)

    if len(chromosome.OS) != expected_operations:
        raise ValueError(
            f"OS length must be {expected_operations}, "
            f"got {len(chromosome.OS)}."
        )

    if len(chromosome.MS) != expected_operations:
        raise ValueError(
            f"MS length must be {expected_operations}, "
            f"got {len(chromosome.MS)}."
        )

    if len(chromosome.WS) != expected_operations:
        raise ValueError(
            f"WS length must be {expected_operations}, "
            f"got {len(chromosome.WS)}."
        )

    for job_id, job_ops in enumerate(instance.jobs):
        actual_count = chromosome.OS.count(job_id)
        expected_count = len(job_ops)

        if actual_count != expected_count:
            raise ValueError(
                f"Job {job_id} must occur {expected_count} times in OS, "
                f"got {actual_count}."
            )

    index = 0
    for job_id, job_ops in enumerate(instance.jobs):
        for operation in job_ops:
            machine_id = chromosome.MS[index]
            worker_id = chromosome.WS[index]

            if machine_id not in operation.compatible_machines:
                raise ValueError(
                    f"Machine {machine_id} is incompatible with "
                    f"operation ({job_id}, {operation.op_id})."
                )

            if worker_id not in operation.compatible_workers:
                raise ValueError(
                    f"Worker {worker_id} is incompatible with "
                    f"operation ({job_id}, {operation.op_id})."
                )

            index += 1