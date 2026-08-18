import math
import os
from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from env.instance_generator import InstanceData


class CPSATSolver:
    """
    CP-SAT solver for the fatigue-free relaxation of FJSP-WF.

    The model preserves:
        1. operation precedence;
        2. machine compatibility and capacity;
        3. worker compatibility and capacity;
        4. worker-dependent skill levels;
        5. minimization of makespan.

    Worker fatigue is fixed at F = 0. Therefore,

        p_ijkq = p_ijk / skill_ijq.

    This problem provides a lower-bound reference for the original
    fatigue-aware scheduling problem because fatigue can only increase
    processing times.

    CP-SAT works with integer time. Floating-point processing times are
    scaled by `time_scale`. Durations are rounded downward so that the
    resulting optimum preserves the lower-bound property.
    """

    def __init__(
        self,
        instance: InstanceData,
        time_limit: float = 3600.0,
        time_scale: int = 1000,
        num_search_workers: Optional[int] = None,
        available_workers: Optional[List[int]] = None,
        log_search_progress: bool = False,
    ):
        self.instance = instance
        self.time_limit = float(time_limit)
        self.time_scale = int(time_scale)

        if self.time_scale <= 0:
            raise ValueError("time_scale must be positive.")

        if num_search_workers is None:
            num_search_workers = min(8, os.cpu_count() or 1)

        self.num_search_workers = int(num_search_workers)
        self.log_search_progress = bool(log_search_progress)

        if available_workers is None:
            self.available_workers = set(range(instance.num_workers))
        else:
            self.available_workers = {int(w) for w in available_workers}

        self.model = cp_model.CpModel()

        self.start_vars: Dict[Tuple[int, int], cp_model.IntVar] = {}
        self.end_vars: Dict[Tuple[int, int], cp_model.IntVar] = {}

        self.presence_vars = {}
        self.optional_intervals = {}
        self.duration_int = {}
        self.duration_float = {}

        self.machine_intervals = {
            m: [] for m in range(instance.num_machines)
        }

        self.worker_intervals = {
            w: [] for w in range(instance.num_workers)
        }

        self._validate_instance()
        self.horizon = self._calculate_horizon()

        self.makespan_var = None


    def _validate_instance(self):
        if not self.available_workers:
            raise ValueError("No available workers.")

        for job_id, job in enumerate(self.instance.jobs):
            for op_id, op in enumerate(job):

                feasible_workers = [
                    w for w in op.compatible_workers
                    if w in self.available_workers
                ]

                if not op.compatible_machines:
                    raise ValueError(
                        f"Operation ({job_id}, {op_id}) has no compatible machine."
                    )

                if not feasible_workers:
                    raise ValueError(
                        f"Operation ({job_id}, {op_id}) has no available compatible worker."
                    )


    def _processing_time(self, job_id, op_id, machine_id, worker_id):
        """
        Fatigue-free processing time:

            p_ijkq = p_ijk / skill_ijq
        """

        op = self.instance.jobs[job_id][op_id]

        base_time = float(op.base_processing_times[machine_id])
        skill = float(op.skill_levels.get(worker_id, 1.0))

        if skill <= 0:
            raise ValueError(
                f"Non-positive skill level for worker {worker_id} "
                f"on operation ({job_id}, {op_id})."
            )

        return base_time / skill


    def _to_integer_duration(self, value: float):
        """
        Floor instead of round-to-nearest.

        This is deliberate: the CP-SAT model is used as a lower-bound
        relaxation, so discretization must not artificially increase
        processing times.
        """
        scaled = math.floor(value * self.time_scale + 1e-9)
        return max(1, scaled)


    def _calculate_horizon(self):
        """
        A safe upper bound for the CP-SAT time horizon.

        Processing every operation sequentially is always a feasible
        temporal upper bound, so summing the longest feasible duration
        of every operation is sufficient.
        """

        horizon = 0

        for job_id, job in enumerate(self.instance.jobs):
            for op_id, op in enumerate(job):

                feasible_durations = []

                for machine_id in op.compatible_machines:
                    for worker_id in op.compatible_workers:

                        if worker_id not in self.available_workers:
                            continue

                        duration = self._processing_time(
                            job_id,
                            op_id,
                            machine_id,
                            worker_id,
                        )

                        feasible_durations.append(
                            self._to_integer_duration(duration)
                        )

                if not feasible_durations:
                    raise RuntimeError(
                        f"No feasible resource pair for ({job_id}, {op_id})."
                    )

                horizon += max(feasible_durations)

        return max(1, horizon)


    def _create_operation_variables(self):
        for job_id, job in enumerate(self.instance.jobs):
            for op_id, op in enumerate(job):

                key = (job_id, op_id)

                start = self.model.new_int_var(
                    0,
                    self.horizon,
                    f"start_J{job_id}_O{op_id}",
                )

                end = self.model.new_int_var(
                    0,
                    self.horizon,
                    f"end_J{job_id}_O{op_id}",
                )

                self.start_vars[key] = start
                self.end_vars[key] = end

                choices = []

                for machine_id in op.compatible_machines:
                    for worker_id in op.compatible_workers:

                        if worker_id not in self.available_workers:
                            continue

                        assign_key = (
                            job_id,
                            op_id,
                            machine_id,
                            worker_id,
                        )

                        presence = self.model.new_bool_var(
                            f"x_J{job_id}_O{op_id}"
                            f"_M{machine_id}_W{worker_id}"
                        )

                        duration_float = self._processing_time(
                            job_id,
                            op_id,
                            machine_id,
                            worker_id,
                        )

                        duration_int = self._to_integer_duration(
                            duration_float
                        )

                        interval = self.model.new_optional_interval_var(
                            start,
                            duration_int,
                            end,
                            presence,
                            f"I_J{job_id}_O{op_id}"
                            f"_M{machine_id}_W{worker_id}",
                        )

                        self.presence_vars[assign_key] = presence
                        self.optional_intervals[assign_key] = interval

                        self.duration_int[assign_key] = duration_int
                        self.duration_float[assign_key] = duration_float

                        self.machine_intervals[machine_id].append(
                            interval
                        )

                        self.worker_intervals[worker_id].append(
                            interval
                        )

                        choices.append(presence)

                self.model.add_exactly_one(choices)


    def _add_precedence_constraints(self):
        for job_id, job in enumerate(self.instance.jobs):

            for op_id in range(len(job) - 1):

                current_key = (job_id, op_id)
                next_key = (job_id, op_id + 1)

                self.model.add(
                    self.start_vars[next_key]
                    >=
                    self.end_vars[current_key]
                )


    def _add_resource_constraints(self):
        for machine_id, intervals in self.machine_intervals.items():
            if intervals:
                self.model.add_no_overlap(intervals)

        for worker_id, intervals in self.worker_intervals.items():
            if intervals:
                self.model.add_no_overlap(intervals)


    def _add_objective(self):
        last_operation_ends = []

        for job_id, job in enumerate(self.instance.jobs):
            last_op_id = len(job) - 1

            last_operation_ends.append(
                self.end_vars[(job_id, last_op_id)]
            )

        self.makespan_var = self.model.new_int_var(
            0,
            self.horizon,
            "makespan",
        )

        self.model.add_max_equality(
            self.makespan_var,
            last_operation_ends,
        )

        self.model.minimize(self.makespan_var)


    def build_model(self):
        self._create_operation_variables()
        self._add_precedence_constraints()
        self._add_resource_constraints()
        self._add_objective()


    def _extract_schedule(self, solver):
        schedule = []

        for (
            job_id,
            op_id,
            machine_id,
            worker_id,
        ), presence in self.presence_vars.items():

            if solver.value(presence) != 1:
                continue

            start = (
                solver.value(self.start_vars[(job_id, op_id)])
                / self.time_scale
            )

            end = (
                solver.value(self.end_vars[(job_id, op_id)])
                / self.time_scale
            )

            assign_key = (
                job_id,
                op_id,
                machine_id,
                worker_id,
            )

            schedule.append({
                "job_id": job_id,
                "op_id": op_id,
                "machine_id": machine_id,
                "worker_id": worker_id,
                "start": float(start),
                "end": float(end),
                "processing_time_cp": (
                    self.duration_int[assign_key]
                    / self.time_scale
                ),
                "processing_time_original": float(
                    self.duration_float[assign_key]
                ),
            })

        schedule.sort(
            key=lambda x: (
                x["start"],
                x["end"],
                x["job_id"],
                x["op_id"],
            )
        )

        return schedule


    def solve(self):
        self.build_model()

        solver = cp_model.CpSolver()

        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_search_workers = self.num_search_workers
        solver.parameters.log_search_progress = self.log_search_progress

        status = solver.solve(self.model)

        status_name = solver.status_name(status)

        result = {
            "status": status_name,
            "is_optimal": status == cp_model.OPTIMAL,
            "time_scale": self.time_scale,
            "time_limit": self.time_limit,
            "wall_time": float(solver.wall_time),
            "num_conflicts": int(solver.num_conflicts),
            "num_branches": int(solver.num_branches),
            "objective": None,
            "best_bound": None,
            "schedule": None,
        }

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):

            result["objective"] = (
                float(solver.objective_value)
                / self.time_scale
            )

            result["best_bound"] = (
                float(solver.best_objective_bound)
                / self.time_scale
            )

            result["schedule"] = self._extract_schedule(solver)

        return result