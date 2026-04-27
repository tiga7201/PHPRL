from dataclasses import dataclass
from typing import List, Dict

import random

import json
from dataclasses import asdict

@dataclass
class Operation:
    job_id: int
    op_id: int
    compatible_machines: List[int]
    compatible_workers: List[int]
    base_processing_times: Dict[int, float]   # machine_id -> p_ijk
    skill_levels: Dict[int, float]            # worker_id -> S_ijq
    difficulty: float = 1.0


@dataclass
class InstanceData:
    num_jobs: int
    num_machines: int
    num_workers: int
    jobs: List[List[Operation]]
    machine_automation: Dict[int, float]
    worker_physical_condition: Dict[int, float]


def create_demo_instance() -> InstanceData:
    jobs = [
        [
            Operation(
                job_id=0,
                op_id=0,
                compatible_machines=[0, 1],
                compatible_workers=[0, 1],
                base_processing_times={0: 6.0, 1: 5.0},
                skill_levels={0: 1.0, 1: 1.0},
                difficulty=1.0,
            ),
            Operation(
                job_id=0,
                op_id=1,
                compatible_machines=[1, 2],
                compatible_workers=[0, 1],
                base_processing_times={1: 7.0, 2: 6.0},
                skill_levels={0: 1.0, 1: 1.0},
                difficulty=1.2,
            ),
        ],
        [
            Operation(
                job_id=1,
                op_id=0,
                compatible_machines=[0, 2],
                compatible_workers=[0, 1],
                base_processing_times={0: 5.0, 2: 8.0},
                skill_levels={0: 1.0, 1: 1.0},
                difficulty=1.1,
            ),
            Operation(
                job_id=1,
                op_id=1,
                compatible_machines=[1, 2],
                compatible_workers=[0, 1],
                base_processing_times={1: 4.0, 2: 5.0},
                skill_levels={0: 1.0, 1: 1.0},
                difficulty=0.9,
            ),
        ],
    ]

    return InstanceData(
        num_jobs=2,
        num_machines=3,
        num_workers=2,
        jobs=jobs,
        machine_automation={0: 0.8, 1: 1.0, 2: 1.2},
        worker_physical_condition={0: 1, 1: 3}
    )


def generate_random_instance(
    num_jobs=None,
    num_machines=None,
    num_workers=None,
    min_ops_per_job=3,
    max_ops_per_job=7,
    proc_time_low=3,
    proc_time_high=30,
    seed=None,
) -> InstanceData:
    """
    Generate a random small-scale FJSP-WF instance.

    This version is designed for early experiments and exact-solver-friendly scales.
    """
    rng = random.Random(seed)

    if num_jobs is None:
        num_jobs = rng.randint(2, 4)
    if num_machines is None:
        num_machines = rng.randint(2, 4)
    if num_workers is None:
        num_workers = rng.randint(2, 4)

    machine_ids = list(range(num_machines))
    worker_ids = list(range(num_workers))

    jobs = []

    for job_id in range(num_jobs):
        num_ops = rng.randint(min_ops_per_job, max_ops_per_job)
        job_ops = []

        for op_id in range(num_ops):

            num_comp_m = rng.randint(1, num_machines)
            num_comp_w = rng.randint(1, num_workers)

            compatible_machines = sorted(rng.sample(machine_ids, num_comp_m))
            compatible_workers = sorted(rng.sample(worker_ids, num_comp_w))

            base_processing_times = {
                m: float(rng.randint(proc_time_low, proc_time_high))
                for m in compatible_machines
            }
            skill_levels = {
                w: 1.0 for w in compatible_workers
            }

            difficulty = float(rng.choice([0.8, 1.0, 1.2]))

            job_ops.append(
                Operation(
                    job_id=job_id,
                    op_id=op_id,
                    compatible_machines=compatible_machines,
                    compatible_workers=compatible_workers,
                    base_processing_times=base_processing_times,
                    skill_levels=skill_levels,
                    difficulty=difficulty,
                )
            )

        jobs.append(job_ops)

    machine_automation = {
        m: float(rng.choice([0.8, 1.0, 1.2]))
        for m in machine_ids
    }
    worker_physical_condition = {
        w: int(rng.choice([1, 2, 3]))
        for w in worker_ids
    }

    return InstanceData(
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_workers=num_workers,
        jobs=jobs,
        machine_automation=machine_automation,
        worker_physical_condition=worker_physical_condition,
    )


def instance_to_dict(instance: InstanceData) -> dict:
    return asdict(instance)


def instance_from_dict(data: dict) -> InstanceData:
    jobs = []
    for job_ops in data["jobs"]:
        ops = []
        for op in job_ops:
            ops.append(
                Operation(
                    job_id=op["job_id"],
                    op_id=op["op_id"],
                    compatible_machines=list(op["compatible_machines"]),
                    compatible_workers=list(op["compatible_workers"]),
                    base_processing_times={int(k): float(v) for k, v in op["base_processing_times"].items()},
                    skill_levels={int(k): float(v) for k, v in op["skill_levels"].items()},
                    difficulty=float(op["difficulty"]),
                )
            )
        jobs.append(ops)

    return InstanceData(
        num_jobs=int(data["num_jobs"]),
        num_machines=int(data["num_machines"]),
        num_workers=int(data["num_workers"]),
        jobs=jobs,
        machine_automation={int(k): float(v) for k, v in data["machine_automation"].items()},
        worker_physical_condition={int(k): float(v) for k, v in data["worker_physical_condition"].items()},
    )


def save_instance_dataset(instances: list[InstanceData], filepath: str):
    payload = [instance_to_dict(inst) for inst in instances]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_instance_dataset(filepath: str) -> list[InstanceData]:
    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [instance_from_dict(item) for item in payload]
