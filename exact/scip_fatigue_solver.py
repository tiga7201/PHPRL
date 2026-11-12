import math
import os
from typing import Dict, List, Optional, Tuple

from pyscipopt import Model, exp, log, quicksum

from env.instance_generator import InstanceData


OperationKey = Tuple[int, int]
ChoiceKey = Tuple[int, int, int, int]


class SCIPFatigueSolver:
    """Global MINLP model for fatigue-aware FJSP with worker assignment."""

    PHYSICAL_CONDITION_RATES = {
        1: (0.05, 0.01),
        2: (0.03, 0.03),
        3: (0.01, 0.05),
    }

    def __init__(
        self,
        instance: InstanceData,
        time_limit: float = 600.0,
        num_threads: Optional[int] = None,
        available_workers: Optional[List[int]] = None,
        initial_fatigue: float = 0.0,
        relative_gap: float = 0.0,
        log_output: bool = False,
    ):
        self.instance = instance
        self.time_limit = float(time_limit)
        self.num_threads = int(
            num_threads if num_threads is not None else min(8, os.cpu_count() or 1)
        )
        self.initial_fatigue = float(initial_fatigue)
        self.relative_gap = float(relative_gap)
        self.log_output = bool(log_output)

        if not 0.0 <= self.initial_fatigue <= 1.0:
            raise ValueError("initial_fatigue must be in [0, 1].")
        if self.time_limit <= 0.0:
            raise ValueError("time_limit must be positive.")
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive.")
        if self.relative_gap < 0.0:
            raise ValueError("relative_gap must be non-negative.")

        if available_workers is None:
            self.available_workers = set(range(instance.num_workers))
        else:
            self.available_workers = {int(worker) for worker in available_workers}

        self.model = Model("fatigue_aware_fjsp")
        self.operations: Dict[OperationKey, object] = {
            (job_id, op_id): operation
            for job_id, job in enumerate(instance.jobs)
            for op_id, operation in enumerate(job)
        }
        self.operation_keys = list(self.operations)

        self.feasible_choices: Dict[OperationKey, List[Tuple[int, int]]] = {}
        self.duration_lb: Dict[OperationKey, float] = {}
        self.duration_ub: Dict[OperationKey, float] = {}

        self.start = {}
        self.end = {}
        self.duration = {}
        self.assignment = {}
        self.machine_assignment = {}
        self.worker_assignment = {}
        self.machine_before = {}
        self.worker_used = {}
        self.worker_source = {}
        self.worker_sink = {}
        self.worker_arc = {}
        self.rest_time = {}
        self.fatigue_start = {}
        self.fatigue_end = {}
        self.makespan = None

        self._validate_and_prepare()
        self.horizon = sum(self.duration_ub.values())
        self.big_m_time = self.horizon
        self.big_m_duration = max(self.duration_ub.values())

    def _validate_and_prepare(self):
        if not self.operations:
            raise ValueError("The instance has no operations.")
        if not self.available_workers:
            raise ValueError("No available workers.")

        fatigue_factor_max = 1.0 + math.log(2.0)

        for key, operation in self.operations.items():
            choices = [
                (int(machine), int(worker))
                for machine in operation.compatible_machines
                for worker in operation.compatible_workers
                if worker in self.available_workers
            ]
            if not choices:
                raise ValueError(f"Operation {key} has no feasible machine-worker pair.")

            base_times = []
            for machine, worker in choices:
                base_time = float(operation.base_processing_times[machine])
                skill = float(operation.skill_levels.get(worker, 1.0))
                if abs(skill - 1.0) > 1e-9:
                    raise ValueError(
                        f"SCIP fatigue model assumes unit skill, but operation {key}, "
                        f"worker {worker} has skill={skill}."
                    )
                if base_time <= 0.0:
                    raise ValueError(f"Operation {key} has non-positive processing time.")
                base_times.append(base_time)

            self.feasible_choices[key] = choices
            self.duration_lb[key] = min(base_times)
            self.duration_ub[key] = max(base_times) * fatigue_factor_max

        for worker in self.available_workers:
            condition = int(round(self.instance.worker_physical_condition[worker]))
            if condition not in self.PHYSICAL_CONDITION_RATES:
                raise ValueError(
                    f"Worker {worker} has invalid physical condition {condition}; "
                    "expected one of 1, 2, 3."
                )

    def _eligible_for_machine(self, key: OperationKey, machine: int) -> bool:
        return any(m == machine for m, _ in self.feasible_choices[key])

    def _eligible_for_worker(self, key: OperationKey, worker: int) -> bool:
        return any(w == worker for _, w in self.feasible_choices[key])

    def _create_core_variables(self):
        for key in self.operation_keys:
            job_id, op_id = key
            suffix = f"J{job_id}_O{op_id}"
            self.start[key] = self.model.addVar(
                f"start_{suffix}", lb=0.0, ub=self.horizon, vtype="C"
            )
            self.end[key] = self.model.addVar(
                f"end_{suffix}", lb=0.0, ub=self.horizon, vtype="C"
            )
            self.duration[key] = self.model.addVar(
                f"duration_{suffix}",
                lb=self.duration_lb[key],
                ub=self.duration_ub[key],
                vtype="C",
            )
            self.model.addCons(
                self.end[key] == self.start[key] + self.duration[key],
                name=f"completion_{suffix}",
            )

            choices = []
            for machine, worker in self.feasible_choices[key]:
                choice_key: ChoiceKey = (job_id, op_id, machine, worker)
                var = self.model.addVar(
                    f"x_{suffix}_M{machine}_W{worker}", vtype="B"
                )
                self.assignment[choice_key] = var
                choices.append(var)
            self.model.addCons(quicksum(choices) == 1, name=f"assign_{suffix}")

        for key in self.operation_keys:
            job_id, op_id = key
            suffix = f"J{job_id}_O{op_id}"
            for machine in range(self.instance.num_machines):
                choice_vars = [
                    self.assignment[(job_id, op_id, m, worker)]
                    for m, worker in self.feasible_choices[key]
                    if m == machine
                ]
                if choice_vars:
                    var = self.model.addVar(f"ym_{suffix}_M{machine}", vtype="B")
                    self.machine_assignment[(key, machine)] = var
                    self.model.addCons(var == quicksum(choice_vars))

            for worker in self.available_workers:
                choice_vars = [
                    self.assignment[(job_id, op_id, machine, w)]
                    for machine, w in self.feasible_choices[key]
                    if w == worker
                ]
                if choice_vars:
                    var = self.model.addVar(f"yw_{suffix}_W{worker}", vtype="B")
                    self.worker_assignment[(key, worker)] = var
                    self.model.addCons(var == quicksum(choice_vars))

    def _add_job_precedence(self):
        for job_id, job in enumerate(self.instance.jobs):
            for op_id in range(len(job) - 1):
                self.model.addCons(
                    self.start[(job_id, op_id + 1)] >= self.end[(job_id, op_id)],
                    name=f"job_order_J{job_id}_O{op_id}_O{op_id + 1}",
                )

    def _add_machine_capacity(self):
        for machine in range(self.instance.num_machines):
            keys = [
                key
                for key in self.operation_keys
                if self._eligible_for_machine(key, machine)
            ]
            for index, first in enumerate(keys):
                for second in keys[index + 1 :]:
                    first_before = self.model.addVar(
                        f"mb_{first[0]}_{first[1]}_{second[0]}_{second[1]}_M{machine}",
                        vtype="B",
                    )
                    second_before = self.model.addVar(
                        f"mb_{second[0]}_{second[1]}_{first[0]}_{first[1]}_M{machine}",
                        vtype="B",
                    )
                    self.machine_before[(first, second, machine)] = first_before
                    self.machine_before[(second, first, machine)] = second_before

                    use_first = self.machine_assignment[(first, machine)]
                    use_second = self.machine_assignment[(second, machine)]
                    self.model.addCons(first_before <= use_first)
                    self.model.addCons(first_before <= use_second)
                    self.model.addCons(second_before <= use_first)
                    self.model.addCons(second_before <= use_second)
                    self.model.addCons(first_before + second_before <= 1)
                    self.model.addCons(
                        first_before + second_before >= use_first + use_second - 1
                    )
                    self.model.addCons(
                        self.start[second]
                        >= self.end[first] - self.big_m_time * (1 - first_before)
                    )
                    self.model.addCons(
                        self.start[first]
                        >= self.end[second] - self.big_m_time * (1 - second_before)
                    )

    def _add_worker_sequences(self):
        for worker in sorted(self.available_workers):
            keys = [
                key
                for key in self.operation_keys
                if self._eligible_for_worker(key, worker)
            ]
            used = self.model.addVar(f"worker_used_W{worker}", vtype="B")
            self.worker_used[worker] = used
            assigned = [self.worker_assignment[(key, worker)] for key in keys]
            self.model.addCons(quicksum(assigned) >= used)
            self.model.addCons(quicksum(assigned) <= len(keys) * used)

            for key in keys:
                source = self.model.addVar(
                    f"source_W{worker}_J{key[0]}_O{key[1]}", vtype="B"
                )
                sink = self.model.addVar(
                    f"sink_W{worker}_J{key[0]}_O{key[1]}", vtype="B"
                )
                self.worker_source[(key, worker)] = source
                self.worker_sink[(key, worker)] = sink

            for first in keys:
                for second in keys:
                    if first == second:
                        continue
                    arc = self.model.addVar(
                        f"wa_W{worker}_{first[0]}_{first[1]}_{second[0]}_{second[1]}",
                        vtype="B",
                    )
                    self.worker_arc[(first, second, worker)] = arc
                    self.model.addCons(arc <= self.worker_assignment[(first, worker)])
                    self.model.addCons(arc <= self.worker_assignment[(second, worker)])
                    self.model.addCons(
                        self.start[second]
                        >= self.end[first] - self.big_m_time * (1 - arc)
                    )

            for key in keys:
                incoming = [
                    self.worker_arc[(other, key, worker)]
                    for other in keys
                    if other != key
                ]
                outgoing = [
                    self.worker_arc[(key, other, worker)]
                    for other in keys
                    if other != key
                ]
                assigned_to_worker = self.worker_assignment[(key, worker)]
                self.model.addCons(
                    self.worker_source[(key, worker)] + quicksum(incoming)
                    == assigned_to_worker
                )
                self.model.addCons(
                    self.worker_sink[(key, worker)] + quicksum(outgoing)
                    == assigned_to_worker
                )

            self.model.addCons(
                quicksum(self.worker_source[(key, worker)] for key in keys) == used
            )
            self.model.addCons(
                quicksum(self.worker_sink[(key, worker)] for key in keys) == used
            )

    def _add_fatigue_dynamics(self):
        for worker in sorted(self.available_workers):
            condition = int(round(self.instance.worker_physical_condition[worker]))
            lam, mu = self.PHYSICAL_CONDITION_RATES[condition]
            keys = [
                key
                for key in self.operation_keys
                if self._eligible_for_worker(key, worker)
            ]

            for key in keys:
                job_id, op_id = key
                assigned = self.worker_assignment[(key, worker)]
                fatigue_start = self.model.addVar(
                    f"fs_J{job_id}_O{op_id}_W{worker}", lb=0.0, ub=1.0, vtype="C"
                )
                fatigue_end = self.model.addVar(
                    f"fe_J{job_id}_O{op_id}_W{worker}", lb=0.0, ub=1.0, vtype="C"
                )
                self.fatigue_start[(key, worker)] = fatigue_start
                self.fatigue_end[(key, worker)] = fatigue_end
                self.model.addCons(fatigue_start <= assigned)
                self.model.addCons(fatigue_end <= assigned)

                source = self.worker_source[(key, worker)]
                self.model.addCons(
                    fatigue_start - self.initial_fatigue <= 1.0 - source
                )
                self.model.addCons(
                    self.initial_fatigue - fatigue_start <= 1.0 - source
                )

                fatigue_after_work = 1.0 - (1.0 - fatigue_start) * exp(
                    -lam * self.duration[key]
                )
                self.model.addCons(
                    fatigue_end - fatigue_after_work <= 1.0 - assigned
                )
                self.model.addCons(
                    fatigue_after_work - fatigue_end <= 1.0 - assigned
                )

            for first in keys:
                for second in keys:
                    if first == second:
                        continue
                    arc = self.worker_arc[(first, second, worker)]
                    rest = self.model.addVar(
                        f"rest_W{worker}_{first[0]}_{first[1]}_{second[0]}_{second[1]}",
                        lb=0.0,
                        ub=self.horizon,
                        vtype="C",
                    )
                    self.rest_time[(first, second, worker)] = rest
                    gap = self.start[second] - self.end[first]
                    self.model.addCons(rest <= self.big_m_time * arc)
                    self.model.addCons(rest <= gap + self.big_m_time * (1 - arc))
                    self.model.addCons(rest >= gap - self.big_m_time * (1 - arc))

                    recovered_fatigue = self.fatigue_end[(first, worker)] * exp(
                        -mu * rest
                    )
                    next_fatigue = self.fatigue_start[(second, worker)]
                    self.model.addCons(next_fatigue - recovered_fatigue <= 1.0 - arc)
                    self.model.addCons(recovered_fatigue - next_fatigue <= 1.0 - arc)

        for key, operation in self.operations.items():
            job_id, op_id = key
            for machine, worker in self.feasible_choices[key]:
                assignment = self.assignment[(job_id, op_id, machine, worker)]
                base_time = float(operation.base_processing_times[machine])
                fatigue = self.fatigue_start[(key, worker)]
                fatigue_duration = base_time * (1.0 + log(1.0 + fatigue))
                self.model.addCons(
                    self.duration[key] - fatigue_duration
                    <= self.big_m_duration * (1 - assignment)
                )
                self.model.addCons(
                    fatigue_duration - self.duration[key]
                    <= self.big_m_duration * (1 - assignment)
                )

    def _add_objective(self):
        self.makespan = self.model.addVar(
            "makespan", lb=0.0, ub=self.horizon, vtype="C"
        )
        for job_id, job in enumerate(self.instance.jobs):
            self.model.addCons(self.makespan >= self.end[(job_id, len(job) - 1)])
        self.model.setObjective(self.makespan, "minimize")

    def build_model(self):
        self._create_core_variables()
        self._add_job_precedence()
        self._add_machine_capacity()
        self._add_worker_sequences()
        self._add_fatigue_dynamics()
        self._add_objective()

    def _extract_schedule(self, solution):
        schedule = []
        for key, operation in self.operations.items():
            job_id, op_id = key
            selected = None
            for machine, worker in self.feasible_choices[key]:
                var = self.assignment[(job_id, op_id, machine, worker)]
                if self.model.getSolVal(solution, var) > 0.5:
                    selected = (machine, worker)
                    break
            if selected is None:
                raise RuntimeError(f"No selected assignment found for operation {key}.")

            machine, worker = selected
            schedule.append(
                {
                    "job_id": job_id,
                    "op_id": op_id,
                    "machine_id": machine,
                    "worker_id": worker,
                    "base_processing_time": float(
                        operation.base_processing_times[machine]
                    ),
                    "start": float(self.model.getSolVal(solution, self.start[key])),
                    "end": float(self.model.getSolVal(solution, self.end[key])),
                    "processing_time": float(
                        self.model.getSolVal(solution, self.duration[key])
                    ),
                    "fatigue_before": float(
                        self.model.getSolVal(
                            solution, self.fatigue_start[(key, worker)]
                        )
                    ),
                    "fatigue_after": float(
                        self.model.getSolVal(
                            solution, self.fatigue_end[(key, worker)]
                        )
                    ),
                }
            )

        schedule.sort(
            key=lambda item: (
                item["start"],
                item["end"],
                item["job_id"],
                item["op_id"],
            )
        )
        return schedule

    def solve(self):
        self.build_model()
        self.model.setParam("limits/time", self.time_limit)
        self.model.setParam("limits/gap", self.relative_gap)
        self.model.setParam("parallel/maxnthreads", self.num_threads)
        if not self.log_output:
            self.model.hideOutput()

        self.model.optimize()

        status = str(self.model.getStatus())
        has_solution = self.model.getNSols() > 0
        is_optimal = status == "optimal"
        objective = None
        best_bound = None
        schedule = None

        if has_solution:
            solution = self.model.getBestSol()
            objective = float(self.model.getSolVal(solution, self.makespan))
            schedule = self._extract_schedule(solution)

        dual_bound = float(self.model.getDualbound())
        if math.isfinite(dual_bound):
            best_bound = dual_bound

        gap = float(self.model.getGap()) if has_solution else None
        if gap is not None and not math.isfinite(gap):
            gap = None

        return {
            "status": status,
            "is_optimal": is_optimal,
            "has_solution": has_solution,
            "objective": objective,
            "best_bound": best_bound,
            "relative_gap": gap,
            "wall_time": float(self.model.getSolvingTime()),
            "time_limit": self.time_limit,
            "num_solutions": int(self.model.getNSols()),
            "num_nodes": int(self.model.getNNodes()),
            "schedule": schedule,
        }
