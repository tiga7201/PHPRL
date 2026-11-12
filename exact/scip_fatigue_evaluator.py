import math
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from env.instance_generator import instance_to_dict


PHYSICAL_CONDITION_RATES = {
    1: (0.05, 0.01),
    2: (0.03, 0.03),
    3: (0.01, 0.05),
}


def _rates(instance, worker_id):
    condition = int(round(instance.worker_physical_condition[worker_id]))
    return PHYSICAL_CONDITION_RATES[condition]


def _solve_in_subprocess(
    instance,
    time_limit_seconds,
    relative_gap,
    num_threads,
    log_output,
):
    """Keep SCIP and PyTorch OpenMP runtimes in separate processes."""
    worker_path = Path(__file__).with_name("scip_fatigue_worker.py")
    if not worker_path.exists():
        raise FileNotFoundError(f"Missing SCIP worker: {worker_path}")

    payload = {
        "instance": instance_to_dict(instance),
        "time_limit_seconds": float(time_limit_seconds),
        "relative_gap": float(relative_gap),
        "num_threads": int(num_threads),
        "log_output": bool(log_output),
    }

    with tempfile.TemporaryDirectory(prefix="phprl_scip_") as temp_dir:
        input_path = Path(temp_dir) / "input.json"
        output_path = Path(temp_dir) / "output.json"
        input_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(worker_path),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
            cwd=str(worker_path.parent.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(time_limit_seconds) + 120.0,
            check=False,
        )

        if log_output and completed.stdout:
            print(completed.stdout, end="")
        if completed.returncode != 0:
            raise RuntimeError(
                "SCIP worker failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )
        if not output_path.exists():
            raise RuntimeError("SCIP worker produced no result file.")
        return json.loads(output_path.read_text(encoding="utf-8"))


def _serial_fallback(instance):
    """Build a deterministic feasible schedule if SCIP has no incumbent."""
    current_time = 0.0
    worker_fatigue = {worker: 0.0 for worker in range(instance.num_workers)}
    worker_last_end = {worker: 0.0 for worker in range(instance.num_workers)}
    schedule = []

    for job_id, job in enumerate(instance.jobs):
        for op_id, operation in enumerate(job):
            machine_id, worker_id = min(
                (
                    (int(machine), int(worker))
                    for machine in operation.compatible_machines
                    for worker in operation.compatible_workers
                ),
                key=lambda pair: (
                    float(operation.base_processing_times[pair[0]]),
                    pair[0],
                    pair[1],
                ),
            )
            lam, mu = _rates(instance, worker_id)
            rest_time = max(0.0, current_time - worker_last_end[worker_id])
            fatigue_before = worker_fatigue[worker_id] * math.exp(-mu * rest_time)
            base_time = float(operation.base_processing_times[machine_id])
            processing_time = base_time * (1.0 + math.log(1.0 + fatigue_before))
            end_time = current_time + processing_time
            fatigue_after = 1.0 - (1.0 - fatigue_before) * math.exp(
                -lam * processing_time
            )

            schedule.append(
                {
                    "job_id": job_id,
                    "op_id": op_id,
                    "machine_id": machine_id,
                    "worker_id": worker_id,
                    "start": current_time,
                    "end": end_time,
                    "processing_time": processing_time,
                    "fatigue_before": fatigue_before,
                    "fatigue_after": fatigue_after,
                }
            )
            worker_fatigue[worker_id] = fatigue_after
            worker_last_end[worker_id] = end_time
            current_time = end_time

    return current_time, schedule


def evaluate_scip_fatigue_method(
    instances,
    seeds,
    time_limit_seconds=600.0,
    relative_gap=0.0,
    num_threads=8,
    log_output=False,
):
    if len(instances) != len(seeds):
        raise ValueError("instances and seeds must have the same length.")

    makespans = []
    best_bounds = []
    solve_times = []
    statuses = []
    optimal_flags = []
    solver_gaps = []
    solution_sources = []
    records = []

    for seed, instance in zip(seeds, instances):
        print(f"    SCIP fatigue seed={seed}")
        try:
            result = _solve_in_subprocess(
                instance=instance,
                time_limit_seconds=time_limit_seconds,
                relative_gap=relative_gap,
                num_threads=num_threads,
                log_output=log_output,
            )
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            print(f"        WARNING: SCIP worker error: {error}")
            result = {
                "status": "worker_error",
                "is_optimal": False,
                "has_solution": False,
                "objective": None,
                "best_bound": None,
                "relative_gap": None,
                "wall_time": float(time_limit_seconds),
                "num_solutions": 0,
                "num_nodes": 0,
                "schedule": None,
            }

        if result["has_solution"]:
            objective = float(result["objective"])
            schedule = result["schedule"]
            source = "scip"
        else:
            objective, schedule = _serial_fallback(instance)
            source = "serial_fallback"
            print(
                "        WARNING: SCIP found no incumbent; "
                f"using deterministic feasible fallback={objective:.4f}"
            )

        bound = result["best_bound"]
        if bound is None or not math.isfinite(float(bound)):
            bound = None
            gap_percent = None
        elif result["is_optimal"]:
            gap_percent = 0.0
        else:
            gap_percent = max(
                0.0,
                (objective - float(bound)) / max(abs(objective), 1e-12) * 100.0,
            )

        makespans.append(objective)
        best_bounds.append(None if bound is None else float(bound))
        solve_times.append(float(result["wall_time"]))
        statuses.append(result["status"])
        optimal_flags.append(bool(result["is_optimal"]))
        solver_gaps.append(gap_percent)
        solution_sources.append(source)
        records.append(
            {
                "seed": int(seed),
                "status": result["status"],
                "is_optimal": bool(result["is_optimal"]),
                "solution_source": source,
                "makespan": objective,
                "best_bound": bound,
                "solver_gap_percent": gap_percent,
                "solve_time_seconds": float(result["wall_time"]),
                "num_solutions": int(result["num_solutions"]),
                "num_nodes": int(result["num_nodes"]),
                "schedule": schedule,
            }
        )

        label = "optimal" if result["is_optimal"] else "best feasible"
        print(
            f"        status={result['status']} | {label}={objective:.4f} | "
            f"bound={bound} | gap={gap_percent} | "
            f"source={source} | time={result['wall_time']:.2f}s"
        )

    num_instances = len(instances)
    num_optimal = sum(optimal_flags)
    num_fallback = sum(source == "serial_fallback" for source in solution_sources)
    optimal_rate = 100.0 * num_optimal / num_instances if num_instances else 0.0

    print(
        "    SCIP fatigue optimality summary | "
        f"optimal={num_optimal}/{num_instances} | "
        f"feasible_not_optimal={num_instances - num_optimal} | "
        f"fallback={num_fallback} | optimal_rate={optimal_rate:.2f}%"
    )

    return {
        "avg_makespan": round(sum(makespans) / len(makespans), 2),
        "worst_makespan": round(max(makespans), 2),
        "avg_solve_time_seconds": round(
            sum(solve_times) / len(solve_times), 4
        ),
        "per_seed": [round(value, 4) for value in makespans],
        "per_seed_best_bound": [
            None if value is None else round(value, 4) for value in best_bounds
        ],
        "per_seed_status": statuses,
        "per_seed_is_optimal": optimal_flags,
        "per_seed_solver_gap_percent": [
            None if value is None else round(value, 4) for value in solver_gaps
        ],
        "per_seed_solution_source": solution_sources,
        "num_instances": num_instances,
        "num_optimal": num_optimal,
        "num_feasible_not_optimal": num_instances - num_optimal,
        "num_fallback": num_fallback,
        "optimal_rate_percent": round(optimal_rate, 2),
        "all_optimal": all(optimal_flags),
        "time_limit_seconds_per_instance": float(time_limit_seconds),
        "relative_gap_limit": float(relative_gap),
        "num_threads": int(num_threads),
        "fatigue_model": "exact_exponential",
        "records": records,
    }
