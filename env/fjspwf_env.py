from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from env.instance_generator import InstanceData, Operation
from env.fatigue import actual_processing_time


Action = Tuple[int, int, int, int]
# (job_id, op_id, machine_id, worker_id)


@dataclass
class ScheduledOp:
    job_id: int
    op_id: int
    machine_id: int
    worker_id: int
    start: float
    end: float
    proc_time: float


class FJSPWFEnv:
    """
    Event-driven environment for FJSP-WF.

    Key semantics:
    - current_time: current decision time
    - valid actions: actions that can start immediately at current_time
    - if no valid action exists and the schedule is not complete,
      the environment automatically advances to the next completion event
    """

    def __init__(self, instance: InstanceData):
        self.instance = instance
        self.reset()

    def reset(self):
        self.current_time = 0.0

        self.machine_available: Dict[int, float] = {
            m: 0.0 for m in range(self.instance.num_machines)
        }
        self.worker_available: Dict[int, float] = {
            w: 0.0 for w in range(self.instance.num_workers)
        }
        self.worker_fatigue: Dict[int, float] = {
            w: 0.0 for w in range(self.instance.num_workers)
        }

        self.worker_workload: Dict[int, float] = {
            w: 0.0 for w in range(self.instance.num_workers)
        }
        self.worker_physical_condition: Dict[int, int] = dict(self.instance.worker_physical_condition)

        # next unscheduled operation index of each job
        self.job_next_op: Dict[int, int] = {
            j: 0 for j in range(self.instance.num_jobs)
        }

        # earliest time that the next operation of each job can start
        # for the first operation, this is 0.0
        self.job_ready_time: Dict[int, float] = {
            j: 0.0 for j in range(self.instance.num_jobs)
        }

        self.schedule: List[ScheduledOp] = []
        self.done = False

        # move to the first genuine decision state
        self._advance_until_decision_or_done()

        return self._get_state()

    def _get_state(self):
        return {
            "time": self.current_time,
            "machine_available": self.machine_available.copy(),
            "worker_available": self.worker_available.copy(),
            "worker_fatigue": self.worker_fatigue.copy(),
            "job_next_op": self.job_next_op.copy(),
            "job_ready_time": self.job_ready_time.copy(),
            "worker_workload": self.worker_workload.copy(),
            "worker_physical_condition": self.worker_physical_condition.copy(),
        }

    def _all_done(self) -> bool:
        for j in range(self.instance.num_jobs):
            if self.job_next_op[j] < len(self.instance.jobs[j]):
                return False
        return True

    def _current_makespan(self) -> float:
        return max((op.end for op in self.schedule), default=0.0)

    def _is_machine_idle_now(self, machine_id: int) -> bool:
        return self.machine_available[machine_id] <= self.current_time + 1e-9

    def _is_worker_idle_now(self, worker_id: int) -> bool:
        return self.worker_available[worker_id] <= self.current_time + 1e-9

    def _is_job_ready_now(self, job_id: int) -> bool:
        return self.job_ready_time[job_id] <= self.current_time + 1e-9

    def get_valid_actions(self) -> List[Action]:
        """
        Return actions that can start immediately at current_time.
        """
        if self.done:
            return []

        actions: List[Action] = []

        for j in range(self.instance.num_jobs):
            next_op_idx = self.job_next_op[j]
            if next_op_idx >= len(self.instance.jobs[j]):
                continue

            if not self._is_job_ready_now(j):
                continue

            op = self.instance.jobs[j][next_op_idx]

            for m in op.compatible_machines:
                if not self._is_machine_idle_now(m):
                    continue

                for w in op.compatible_workers:
                    if not self._is_worker_idle_now(w):
                        continue
                    actions.append((j, next_op_idx, m, w))

        return actions

    def _get_running_end_times_after_now(self) -> List[float]:
        """
        Collect completion times strictly greater than current_time
        from busy machines and workers.
        """
        candidates = []

        for t in self.machine_available.values():
            if t > self.current_time + 1e-9:
                candidates.append(t)

        for t in self.worker_available.values():
            if t > self.current_time + 1e-9:
                candidates.append(t)

        return candidates

    def _advance_to_next_event(self):
        candidates = self._get_running_end_times_after_now()
        if not candidates:
            raise RuntimeError(
                "No valid action and no future event found. "
                "Environment state is inconsistent."
            )

        old_time = self.current_time
        new_time = min(candidates)
        delta_time = new_time - old_time

        # resting recovery for idle workers during the advanced interval
        for w in range(self.instance.num_workers):
            rest_start = max(old_time, self.worker_available[w])
            rest_duration = new_time - rest_start

            if rest_duration > 1e-9:
                fatigue_before = self.worker_fatigue[w]
                f_p = self.worker_physical_condition[w]
                fatigue_after = self._rollout_fatigue_with_pgnn(
                    fatigue=fatigue_before,
                    f_p=f_p,
                    status=0,
                    duration=rest_duration,
                )
                self.worker_fatigue[w] = fatigue_after

        self.current_time = new_time

    def _advance_until_decision_or_done(self):
        """
        Repeatedly advance time until:
        - there exists at least one valid action, or
        - the schedule is complete
        """
        while True:
            self.done = self._all_done()
            if self.done:
                return

            valid_actions = self.get_valid_actions()
            if len(valid_actions) > 0:
                return

            self._advance_to_next_event()

    def step(self, action: Action):
        if self.done:
            raise ValueError("Environment is already done.")

        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            raise ValueError(
                f"Invalid action at current_time={self.current_time}: {action}. "
                f"Valid actions are: {valid_actions}"
            )

        job_id, op_id, machine_id, worker_id = action
        prev_makespan = self._current_makespan()
        op: Operation = self.instance.jobs[job_id][op_id]

        # In event-driven setting, the chosen action starts immediately
        start_time = self.current_time

        base_time = op.base_processing_times[machine_id]
        fatigue_before = self.worker_fatigue[worker_id]

        proc_time = actual_processing_time(base_time, fatigue_before)
        end_time = start_time + proc_time

        # occupy resources until end_time
        self.machine_available[machine_id] = end_time
        self.worker_available[worker_id] = end_time

        # update job precedence
        self.job_ready_time[job_id] = end_time
        self.job_next_op[job_id] += 1

        # simplified fatigue update for now
        f_p = self.worker_physical_condition[worker_id]
        # fatigue_after = self._rollout_fatigue_with_pgnn(
        #     fatigue=fatigue_before,
        #     f_p=f_p,
        #     status=1,
        #     duration=proc_time,
        # )
        difficulty = op.difficulty
        automation = self.instance.machine_automation[machine_id]
        workload_before = self.worker_workload[worker_id]

        fatigue_after = self._rollout_fatigue_with_pgnn(
            fatigue=fatigue_before,
            f_p=f_p,
            status=1,
            duration=proc_time,
            difficulty=difficulty,
            automation=automation,
            workload=workload_before,
        )
        self.worker_fatigue[worker_id] = fatigue_after
        self.worker_workload[worker_id] += proc_time

        self.schedule.append(
            ScheduledOp(
                job_id=job_id,
                op_id=op_id,
                machine_id=machine_id,
                worker_id=worker_id,
                start=start_time,
                end=end_time,
                proc_time=proc_time,
            )
        )

        # After executing one action:
        # - if there are still valid actions at the same current_time, stay here
        # - otherwise advance automatically until the next decision state or done
        self._advance_until_decision_or_done()

        makespan = self._current_makespan()
        reward = -(makespan - prev_makespan)

        next_state = self._get_state()
        info = {
            "makespan": makespan,
            "action_start_time": start_time,
            "action_end_time": end_time,
            "proc_time": proc_time,
            "fatigue_before": fatigue_before,
            "fatigue_after": fatigue_after,
            "valid_actions_next": self.get_valid_actions(),
        }

        return next_state, reward, self.done, info

    def load_pgnn_phase1(self, checkpoint_path: str, device: str = "cpu"):
        from utils.pgnn_inference import PGNNPhase1Inference
        self.pgnn_phase1 = PGNNPhase1Inference(checkpoint_path, device=device)

    def load_pgnn_phase2(self, checkpoint_path: str, device: str = "cpu"):
        from utils.pgnn_inference import PGNNPhase2Inference
        self.pgnn_phase2 = PGNNPhase2Inference(checkpoint_path, device=device)

    def _rollout_fatigue_with_pgnn(
            self,
            fatigue: float,
            f_p: int,
            status: int,
            duration: float,
            difficulty: float = 1.0,
            automation: float = 1.0,
            workload: float = 0.0,
    ) -> float:
        if not hasattr(self, "pgnn_phase1"):
            return fatigue

        whole_steps = int(duration)
        frac = duration - whole_steps

        def one_step_update(fatigue_value: float, frac_scale: float = 1.0) -> float:
            delta_f_1 = self.pgnn_phase1.predict_delta_f(
                fatigue=fatigue_value,
                f_p=f_p,
                status=status,
            )

            if status == 1 and hasattr(self, "pgnn_phase2"):
                delta_f = self.pgnn_phase2.predict_delta_f(
                    delta_f_1=delta_f_1,
                    difficulty=difficulty,
                    automation=automation,
                    workload=workload,
                    fatigue=fatigue_value,
                    status=status,
                )
            else:
                delta_f = delta_f_1

            new_fatigue = fatigue_value + frac_scale * delta_f
            new_fatigue = max(0.0, min(1.0, new_fatigue))
            return new_fatigue

        for _ in range(whole_steps):
            fatigue = one_step_update(fatigue, 1.0)

        if frac > 1e-9:
            fatigue = one_step_update(fatigue, frac)

        return fatigue

    def _compute_proc_time_for_action(self, base_time: float, fatigue: float, skill: float = 1.0) -> float:
        return actual_processing_time(base_time, fatigue, skill)
