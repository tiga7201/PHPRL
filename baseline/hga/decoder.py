from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from env.fjspwf_env import FJSPWFEnv, ScheduledOp
from env.instance_generator import InstanceData
from utils.pgnn_inference import (
    PGNNPhase1Inference,
    PGNNPhase2Inference,
)

from .chromosome import (
    Chromosome,
    global_operation_index,
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
    ):
        self.instance = instance
        self.device = device
        self.use_fatigue = bool(use_fatigue)

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
        )

        # PGNN inference objects are stateless during evaluation and can be
        # shared by every chromosome evaluation.
        if self.pgnn_phase1 is not None:
            env.pgnn_phase1 = self.pgnn_phase1

        if self.pgnn_phase2 is not None:
            env.pgnn_phase2 = self.pgnn_phase2

        return env

    def decode(
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
    chromosome: Chromosome,
    instance: InstanceData,
    phase1_checkpoint: Optional[str] = "checkpoints/pgnn_phase1.pt",
    phase2_checkpoint: Optional[str] = None,
    device: str = "cpu",
    available_workers: Optional[Sequence[int]] = None,
    use_fatigue: bool = True,
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
    )
    return decoder.decode(chromosome)
