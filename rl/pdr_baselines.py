from typing import List, Tuple

Action = Tuple[int, int, int, int]  # (job_id, op_id, machine_id, worker_id)


def _get_action_ready_time(env, action: Action) -> float:
    job_id, op_id, machine_id, worker_id = action
    job_ready = env.job_ready_time[job_id]
    machine_ready = env.machine_available[machine_id]
    worker_ready = env.worker_available[worker_id]
    return max(job_ready, machine_ready, worker_ready)


def _get_action_actual_proc_time(env, action: Action) -> float:
    job_id, op_id, machine_id, worker_id = action
    op = env.instance.jobs[job_id][op_id]
    base_time = op.base_processing_times[machine_id]
    fatigue = env.worker_fatigue[worker_id]
    skill = 1.0
    return env._compute_proc_time_for_action(base_time, fatigue, skill)


def _estimate_job_remaining_work(env, job_id: int) -> float:
    """
    Estimate remaining work of a job from current next operation onward.
    For each remaining operation, use the minimum actual processing time
    among its compatible machine-worker pairs under current fatigue.
    """
    remaining_work = 0.0
    next_op_id = env.job_next_op[job_id]
    job_ops = env.instance.jobs[job_id]

    for op_id in range(next_op_id, len(job_ops)):
        op = job_ops[op_id]
        candidate_times = []

        for m in op.compatible_machines:
            for w in op.compatible_workers:
                base_time = op.base_processing_times[m]
                fatigue = env.worker_fatigue[w]
                skill = 1.0
                proc_time = env._compute_proc_time_for_action(base_time, fatigue, skill)
                candidate_times.append(proc_time)

        if candidate_times:
            remaining_work += min(candidate_times)

    return remaining_work


def select_action_fifo(env) -> Action:
    """
    FIFO extension to O-M-W triplets:
    1) prioritize the action whose operation becomes ready earliest
    2) tie-break by job_id, op_id, machine_id, worker_id
    """
    valid_actions = env.get_valid_actions()
    if not valid_actions:
        raise RuntimeError("No valid action available for FIFO.")

    scored = []
    for a in valid_actions:
        ready_time = _get_action_ready_time(env, a)
        job_id, op_id, machine_id, worker_id = a
        scored.append((ready_time, job_id, op_id, machine_id, worker_id, a))

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    return scored[0][-1]


def select_action_spt(env) -> Action:
    """
    SPT extension to O-M-W triplets:
    directly choose the feasible triplet with the shortest actual processing time.
    """
    valid_actions = env.get_valid_actions()
    if not valid_actions:
        raise RuntimeError("No valid action available for SPT.")

    scored = []
    for a in valid_actions:
        proc_time = _get_action_actual_proc_time(env, a)
        job_id, op_id, machine_id, worker_id = a
        scored.append((proc_time, job_id, op_id, machine_id, worker_id, a))

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    return scored[0][-1]


def select_action_mwkr(env) -> Action:
    """
    MWKR extension to O-M-W triplets:
    choose a feasible triplet whose corresponding job has the most work remaining.
    Tie-break by earlier ready time, then ids.
    """
    valid_actions = env.get_valid_actions()
    if not valid_actions:
        raise RuntimeError("No valid action available for MWKR.")

    scored = []
    for a in valid_actions:
        job_id, op_id, machine_id, worker_id = a
        remaining_work = _estimate_job_remaining_work(env, job_id)
        ready_time = _get_action_ready_time(env, a)

        # negative remaining_work because we want descending remaining work
        scored.append((-remaining_work, ready_time, job_id, op_id, machine_id, worker_id, a))

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))
    return scored[0][-1]


def select_pdr_action(env, rule: str) -> Action:
    rule = rule.upper()

    if rule == "FIFO":
        return select_action_fifo(env)
    elif rule == "SPT":
        return select_action_spt(env)
    elif rule == "MWKR":
        return select_action_mwkr(env)
    else:
        raise ValueError(f"Unknown PDR rule: {rule}")