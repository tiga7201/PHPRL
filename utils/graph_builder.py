from typing import Dict, List, Tuple
import numpy as np

from env.fjspwf_env import FJSPWFEnv
from env.fatigue import actual_processing_time


def build_hypergraph_state(env: FJSPWFEnv) -> Dict:
    """
    Build hypergraph-style state representation from the current environment state.

    Returns a dict with:
    - op_features: [num_ops, 7]
    - machine_features: [num_machines, 5]
    - worker_features: [num_workers, 7]
    - edge_features: [num_edges, 3]
    - edge_links: list of (global_op_id, machine_id, worker_id)
    - valid_action_mask: [num_edges]
    - action_to_edge: dict[action] -> edge_idx
    """
    instance = env.instance

    op_features: List[List[float]] = []
    machine_features: List[List[float]] = []
    worker_features: List[List[float]] = []
    edge_features: List[List[float]] = []
    edge_links: List[Tuple[int, int, int]] = []
    action_to_edge = {}

    # build global op index mapping
    op_id_map = {}
    reverse_op_id_map = {}
    global_op_idx = 0
    for j in range(instance.num_jobs):
        for op in instance.jobs[j]:
            op_id_map[(op.job_id, op.op_id)] = global_op_idx
            reverse_op_id_map[global_op_idx] = (op.job_id, op.op_id)
            global_op_idx += 1

    total_ops = global_op_idx

    # -------- operation node features --------
    for j in range(instance.num_jobs):
        job_ops = instance.jobs[j]
        num_unscheduled = len(job_ops) - env.job_next_op[j]

        # estimated job completion time
        est_job_completion = env.job_ready_time[j]
        for future_op_idx in range(env.job_next_op[j], len(job_ops)):
            future_op = job_ops[future_op_idx]
            avg_p = np.mean(list(future_op.base_processing_times.values()))
            est_job_completion += avg_p

        for op in job_ops:
            scheduled = 1.0 if op.op_id < env.job_next_op[j] else 0.0

            # number of connected hyperedges
            num_edges = len(op.compatible_machines) * len(op.compatible_workers)

            if scheduled:
                # find actual scheduled record
                matched = None
                for item in env.schedule:
                    if item.job_id == op.job_id and item.op_id == op.op_id:
                        matched = item
                        break

                proc_time = matched.proc_time if matched else 0.0
                start_time = matched.start if matched else 0.0
            else:
                proc_time = float(np.mean(list(op.base_processing_times.values())))

                if op.op_id == 0:
                    start_time = 0.0
                else:
                    if op.op_id < env.job_next_op[j]:
                        prev_sched = None
                        for item in env.schedule:
                            if item.job_id == op.job_id and item.op_id == op.op_id - 1:
                                prev_sched = item
                                break
                        start_time = prev_sched.end if prev_sched else 0.0
                    else:
                        # for unscheduled operations, use current job ready time as a rough estimate
                        start_time = env.job_ready_time[j]

            op_features.append([
                scheduled,                  # 1 status
                float(num_edges),           # 2 number of hyperedges
                float(num_unscheduled),     # 3 unscheduled ops in job
                float(proc_time),           # 4 processing time
                float(start_time),          # 5 start time
                float(est_job_completion),  # 6 job completion time
                float(op.difficulty),       # 7 processing difficulty
            ])

    # -------- machine node features --------
    elapsed_time = max(env.current_time, 1e-6)

    for m in range(instance.num_machines):
        busy = 1.0 if env.machine_available[m] > env.current_time + 1e-9 else 0.0

        num_edges = 0
        busy_time = 0.0
        for j in range(instance.num_jobs):
            for op in instance.jobs[j]:
                if m in op.compatible_machines:
                    num_edges += len(op.compatible_workers)

        for item in env.schedule:
            if item.machine_id == m:
                busy_time += item.proc_time

        utilization = busy_time / max(env.current_time, 1.0)

        machine_features.append([
            busy,                                        # 1 status
            float(num_edges),                            # 2 number of hyperedges
            float(env.machine_available[m]),             # 3 available time
            float(utilization),                          # 4 utilization
            float(instance.machine_automation[m]),       # 5 automation level
        ])

    # -------- worker node features --------
    for w in range(instance.num_workers):
        busy = 1.0 if env.worker_available[w] > env.current_time + 1e-9 else 0.0

        num_edges = 0
        busy_time = 0.0
        workload = 0.0

        for j in range(instance.num_jobs):
            for op in instance.jobs[j]:
                if w in op.compatible_workers:
                    num_edges += len(op.compatible_machines)

        for item in env.schedule:
            if item.worker_id == w:
                busy_time += item.proc_time
                workload += item.proc_time

        utilization = busy_time / max(env.current_time, 1.0)

        worker_features.append([
            busy,                                            # 1 status
            float(num_edges),                                # 2 number of hyperedges
            float(env.worker_available[w]),                  # 3 available time
            float(utilization),                              # 4 utilization
            float(instance.worker_physical_condition[w]),    # 5 physical condition
            float(workload),                                 # 6 current workload
            float(env.worker_fatigue[w]),                    # 7 fatigue
        ])

    # -------- hyperedges --------
    valid_actions = set(env.get_valid_actions())

    for j in range(instance.num_jobs):
        for op in instance.jobs[j]:
            global_idx = op_id_map[(op.job_id, op.op_id)]

            for m in op.compatible_machines:
                for w in op.compatible_workers:
                    base_time = op.base_processing_times[m]
                    skill = op.skill_levels[w]
                    fatigue = env.worker_fatigue[w]
                    proc_time = actual_processing_time(base_time, fatigue, skill)

                    edge_idx = len(edge_features)
                    edge_features.append([
                        float(base_time),   # 1 standard processing time
                        float(skill),       # 2 skill level
                        float(proc_time),   # 3 actual processing time
                    ])
                    edge_links.append((global_idx, m, w))

                    action = (op.job_id, op.op_id, m, w)
                    action_to_edge[action] = edge_idx

    valid_action_mask = np.zeros(len(edge_features), dtype=np.float32)
    for action in valid_actions:
        edge_idx = action_to_edge[action]
        valid_action_mask[edge_idx] = 1.0

    edge_to_action = {idx: action for action, idx in action_to_edge.items()}

    return {
        "op_features": np.array(op_features, dtype=np.float32),
        "machine_features": np.array(machine_features, dtype=np.float32),
        "worker_features": np.array(worker_features, dtype=np.float32),
        "edge_features": np.array(edge_features, dtype=np.float32),
        "edge_links": np.array(edge_links, dtype=np.int64),
        "valid_action_mask": valid_action_mask,
        "action_to_edge": action_to_edge,
        "edge_to_action": edge_to_action,
        "op_id_map": op_id_map,
        "reverse_op_id_map": reverse_op_id_map,
    }
