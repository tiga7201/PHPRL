from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from env.fatigue import actual_processing_time
from env.fjspwf_env import FJSPWFEnv, ScheduledOp
from env.instance_generator import InstanceData
from utils.pgnn_inference import (
    PGNNPhase1Inference,
    PGNNPhase2Inference,
)

from .chromosome import (
    Chromosome,
    global_operation_index,
    operation_offsets,
    validate_chromosome,
)


class HgaDecoder:
    """
    Decode OS-MS-WS chromosomes with the existing event-driven environment.

    OS acts as a priority list. At each decision time, the first schedulable
    operation in OS is selected. MS and WS determine its fixed machine and
    worker assignment.
    """

    def __init__(
        self,
        instance: InstanceData,
        phase1_checkpoint: Optional[str] = "checkpoints/pgnn_phase1.pt",
        phase2_checkpoint: Optional[str] = None,
        device: str = "cpu",
        available_workers: Optional[Sequence[int]] = None,
        use_fatigue: bool = True,
        use_fast_decoder: bool = True,
    ):
        self.instance = instance
        self.device = device
        self.use_fatigue = bool(use_fatigue)
        self.use_fast_decoder = bool(use_fast_decoder)

        if available_workers is None:
            self.available_workers = None
        else:
            self.available_workers = tuple(int(w) for w in available_workers)

        self.pgnn_phase1 = None
        self.pgnn_phase2 = None

        if self.use_fatigue:
            if phase1_checkpoint is None:
                raise ValueError(
                    "phase1_checkpoint is required when use_fatigue=True."
                )

            phase1_path = Path(phase1_checkpoint)
            if not phase1_path.is_file():
                raise FileNotFoundError(
                    f"PGNN phase-1 checkpoint not found: {phase1_path}"
                )

            self.pgnn_phase1 = PGNNPhase1Inference(
                str(phase1_path),
                device=device,
            )

            if phase2_checkpoint is not None:
                phase2_path = Path(phase2_checkpoint)
                if not phase2_path.is_file():
                    raise FileNotFoundError(
                        f"PGNN phase-2 checkpoint not found: {phase2_path}"
                    )

                self.pgnn_phase2 = PGNNPhase2Inference(
                    str(phase2_path),
                    device=device,
                )

    def _make_env(self) -> FJSPWFEnv:
        env = FJSPWFEnv(
            instance=self.instance,
            available_workers=self.available_workers,
            use_fatigue=self.use_fatigue,
            record_fatigue_traces=False,
        )

        # PGNN inference objects are stateless during evaluation and can be
        # shared by every chromosome evaluation.
        if self.pgnn_phase1 is not None:
            env.pgnn_phase1 = self.pgnn_phase1

        if self.pgnn_phase2 is not None:
            env.pgnn_phase2 = self.pgnn_phase2

        return env

    def _rollout_fatigue(
        self,
        fatigue: float,
        f_p: int,
        status: int,
        duration: float,
        difficulty: float = 1.0,
        automation: float = 1.0,
        workload: float = 0.0,
        start_time: float = 0.0,
    ) -> float:
        if self.pgnn_phase1 is None:
            return fatigue

        current_workload = float(workload)
        current_time = float(start_time)
        remaining = float(duration)

        while remaining > 1e-9:
            if abs(current_time - round(current_time)) < 1e-9:
                next_boundary = current_time + 1.0
            else:
                next_boundary = float(int(current_time) + 1)

            step_len = min(remaining, next_boundary - current_time)
            delta_f = self.pgnn_phase1.predict_delta_f(
                fatigue=fatigue,
                f_p=f_p,
                status=status,
            )

            if status == 1 and self.pgnn_phase2 is not None:
                delta_f = self.pgnn_phase2.predict_delta_f(
                    delta_f_1=delta_f,
                    difficulty=difficulty,
                    automation=automation,
                    workload=current_workload,
                    fatigue=fatigue,
                    status=status,
                )

            fatigue = max(0.0, min(1.0, fatigue + step_len * delta_f))

            if status == 1:
                current_workload += step_len

            current_time += step_len
            remaining -= step_len

        return fatigue

    def _decode_fast(
        self,
        chromosome: Chromosome,
    ) -> Tuple[List[ScheduledOp], float]:
        validate_chromosome(chromosome, self.instance)

        num_jobs = self.instance.num_jobs
        num_machines = self.instance.num_machines
        num_workers = self.instance.num_workers
        offsets = operation_offsets(self.instance)

        machine_available = [0.0] * num_machines
        worker_available = [0.0] * num_workers
        worker_fatigue = [0.0] * num_workers
        worker_workload = [0.0] * num_workers
        worker_physical = [
            self.instance.worker_physical_condition[w]
            for w in range(num_workers)
        ]
        job_next_op = [0] * num_jobs
        job_ready_time = [0.0] * num_jobs

        if self.available_workers is None:
            available_workers = set(range(num_workers))
        else:
            available_workers = set(self.available_workers)

        occurrence_count = [0] * num_jobs
        position_op_ids = []
        position_gene_indices = []
        for job_id in chromosome.OS:
            op_id = occurrence_count[job_id]
            occurrence_count[job_id] += 1
            position_op_ids.append(op_id)
            position_gene_indices.append(offsets[job_id] + op_id)

        remaining_positions = list(range(len(chromosome.OS)))
        schedule: List[ScheduledOp] = []
        current_time = 0.0

        while remaining_positions:
            selected_list_index = None

            for list_index, position in enumerate(remaining_positions):
                job_id = chromosome.OS[position]
                op_id = position_op_ids[position]

                if job_next_op[job_id] != op_id:
                    continue
                if job_ready_time[job_id] > current_time + 1e-9:
                    continue

                gene_index = position_gene_indices[position]
                machine_id = int(chromosome.MS[gene_index])
                worker_id = int(chromosome.WS[gene_index])

                if worker_id not in available_workers:
                    continue
                if machine_available[machine_id] > current_time + 1e-9:
                    continue
                if worker_available[worker_id] > current_time + 1e-9:
                    continue

                selected_list_index = list_index
                break

            if selected_list_index is None:
                next_event = None

                for end_time in machine_available:
                    if end_time > current_time + 1e-9:
                        next_event = (
                            end_time
                            if next_event is None
                            else min(next_event, end_time)
                        )

                for end_time in worker_available:
                    if end_time > current_time + 1e-9:
                        next_event = (
                            end_time
                            if next_event is None
                            else min(next_event, end_time)
                        )

                if next_event is None:
                    raise RuntimeError(
                        "Chromosome decoding reached a deadlock at "
                        f"time={current_time}. Remaining OS positions: "
                        f"{remaining_positions[:10]}"
                    )

                old_time = current_time
                current_time = next_event

                if self.use_fatigue:
                    for worker_id in range(num_workers):
                        rest_start = max(
                            old_time,
                            worker_available[worker_id],
                        )
                        rest_duration = current_time - rest_start

                        if rest_duration > 1e-9:
                            worker_fatigue[worker_id] = self._rollout_fatigue(
                                fatigue=worker_fatigue[worker_id],
                                f_p=worker_physical[worker_id],
                                status=0,
                                duration=rest_duration,
                                start_time=rest_start,
                            )

                continue

            position = remaining_positions.pop(selected_list_index)
            job_id = chromosome.OS[position]
            op_id = position_op_ids[position]
            gene_index = position_gene_indices[position]
            machine_id = int(chromosome.MS[gene_index])
            worker_id = int(chromosome.WS[gene_index])
            operation = self.instance.jobs[job_id][op_id]

            base_time = operation.base_processing_times[machine_id]
            fatigue_before = worker_fatigue[worker_id]
            skill = operation.skill_levels.get(worker_id, 1.0)

            if self.use_fatigue:
                proc_time = actual_processing_time(
                    base_time,
                    fatigue_before,
                    skill,
                )
            else:
                proc_time = base_time / skill

            start_time = current_time
            end_time = start_time + proc_time

            machine_available[machine_id] = end_time
            worker_available[worker_id] = end_time
            job_ready_time[job_id] = end_time
            job_next_op[job_id] += 1

            if self.use_fatigue:
                worker_fatigue[worker_id] = self._rollout_fatigue(
                    fatigue=fatigue_before,
                    f_p=worker_physical[worker_id],
                    status=1,
                    duration=proc_time,
                    difficulty=operation.difficulty,
                    automation=self.instance.machine_automation[machine_id],
                    workload=worker_workload[worker_id],
                    start_time=start_time,
                )

            worker_workload[worker_id] += proc_time
            schedule.append(
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

        makespan = max((op.end for op in schedule), default=0.0)
        chromosome.fitness = makespan
        return schedule, makespan

    def _decode_with_environment(
        self,
        chromosome: Chromosome,
    ) -> Tuple[List[ScheduledOp], float]:
        validate_chromosome(chromosome, self.instance)

        env = self._make_env()
        remaining_positions = list(range(len(chromosome.OS)))

        while remaining_positions:
            valid_actions = set(env.get_valid_actions())

            if env.done:
                break

            selected_position = None
            selected_action = None

            for position in remaining_positions:
                job_id = chromosome.OS[position]

                # The occurrence number determines the operation ID.
                op_id = chromosome.OS[:position].count(job_id)

                # A later operation of the same job cannot be selected before
                # its predecessor has been completed.
                if env.job_next_op[job_id] != op_id:
                    continue

                gene_index = global_operation_index(
                    self.instance,
                    job_id,
                    op_id,
                )
                machine_id = chromosome.MS[gene_index]
                worker_id = chromosome.WS[gene_index]

                action = (
                    job_id,
                    op_id,
                    machine_id,
                    worker_id,
                )

                if action in valid_actions:
                    selected_position = position
                    selected_action = action
                    break

            if selected_action is not None:
                env.step(selected_action)
                remaining_positions.remove(selected_position)
                continue

            # Valid environment actions may exist only for machine-worker
            # assignments different from those encoded by this chromosome.
            # In that case, preserve the chromosome assignment and wait for
            # the next resource-completion event.
            future_events = env._get_running_end_times_after_now()

            if not future_events:
                raise RuntimeError(
                    "Chromosome decoding reached a deadlock at "
                    f"time={env.current_time}. Remaining OS positions: "
                    f"{remaining_positions[:10]}"
                )

            env._advance_to_next_event()

        if remaining_positions:
            raise RuntimeError(
                "Environment finished before all chromosome genes "
                "were decoded."
            )

        schedule = list(env.schedule)
        makespan = max(
            (scheduled_op.end for scheduled_op in schedule),
            default=0.0,
        )

        chromosome.fitness = makespan
        return schedule, makespan

    def decode(
        self,
        chromosome: Chromosome,
    ) -> Tuple[List[ScheduledOp], float]:
        if self.use_fast_decoder:
            return self._decode_fast(chromosome)
        return self._decode_with_environment(chromosome)


def decode(
    chromosome: Chromosome,
    instance: InstanceData,
    phase1_checkpoint: Optional[str] = "checkpoints/pgnn_phase1.pt",
    phase2_checkpoint: Optional[str] = None,
    device: str = "cpu",
    available_workers: Optional[Sequence[int]] = None,
    use_fatigue: bool = True,
    use_fast_decoder: bool = True,
) -> Tuple[List[ScheduledOp], float]:
    """
    Convenience interface for decoding one chromosome.

    During GA optimization, create one HgaDecoder and reuse it instead of
    calling this function repeatedly.
    """
    decoder = HgaDecoder(
        instance=instance,
        phase1_checkpoint=phase1_checkpoint,
        phase2_checkpoint=phase2_checkpoint,
        device=device,
        available_workers=available_workers,
        use_fatigue=use_fatigue,
        use_fast_decoder=use_fast_decoder,
    )
    return decoder.decode(chromosome)
