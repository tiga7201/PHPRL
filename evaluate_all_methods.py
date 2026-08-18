import os
import json
import math
import time

import torch

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state

from models.actor_shyper_full import SHyperActorFull
from models.q_critic_shyper_full import SHyperQCriticFull
from rl.sac_agent import SACAgent

from rl.pdr_baselines import select_pdr_action

# CP-SAT exact solver
from exact.cpsat_solver import CPSATSolver


# ============================================================
# PGNN checkpoint
# ============================================================

def get_pgnn_checkpoint_path():
    project_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        project_root,
        "checkpoints",
        "pgnn_phase1.pt",
    )


# ============================================================
# Instance generation
# ============================================================

def build_instances(
    seeds,
    num_jobs,
    num_machines,
    num_workers,
    min_ops_per_job,
    max_ops_per_job,
):
    """
    Generate each test instance only once.

    All methods use exactly the same InstanceData objects.
    """

    instances = []

    for seed in seeds:

        instance = generate_random_instance(
            seed=seed,
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_workers=num_workers,
            min_ops_per_job=min_ops_per_job,
            max_ops_per_job=max_ops_per_job,
        )

        # ----------------------------------------------------
        # This study does not consider worker skill differences.
        # All skill levels must therefore be equal to 1.
        # ----------------------------------------------------
        for job in instance.jobs:
            for op in job:
                for worker_id in op.compatible_workers:

                    skill = float(
                        op.skill_levels.get(worker_id, 1.0)
                    )

                    if abs(skill - 1.0) > 1e-9:
                        raise ValueError(
                            f"Non-unit skill detected: "
                            f"seed={seed}, "
                            f"job={op.job_id}, "
                            f"op={op.op_id}, "
                            f"worker={worker_id}, "
                            f"skill={skill}"
                        )

        instances.append(instance)

    return instances


# ============================================================
# Environment
# ============================================================

def make_env(
    instance,
    use_fatigue: bool,
):
    """
    Build an environment from an already generated instance.

    use_fatigue = True:
        PGNN fatigue evolution is enabled.

    use_fatigue = False:
        fatigue is disabled and processing time equals
        the base processing time because skill = 1.
    """

    env = FJSPWFEnv(
        instance=instance,
        use_fatigue=use_fatigue,
    )

    # PGNN is only needed when fatigue is enabled.
    if use_fatigue:

        pgnn_path = get_pgnn_checkpoint_path()

        if not os.path.exists(pgnn_path):
            raise FileNotFoundError(
                f"PGNN checkpoint not found: {pgnn_path}"
            )

        env.load_pgnn_phase1(
            pgnn_path,
            device="cpu",
        )

    return env


# ============================================================
# RL evaluation
# ============================================================

def run_episode(
    env,
    agent,
):
    env.reset()

    done = False
    last_info = None

    start_time = time.perf_counter()

    while not done:

        graph_state = build_hypergraph_state(env)

        decision = agent.actor.select_greedy_action(
            graph_state
        )

        action = decision["action"]

        _, _, done, info = env.step(action)

        last_info = info

    solve_time = (
        time.perf_counter()
        - start_time
    )

    return (
        float(last_info["makespan"]),
        float(solve_time),
    )


# ============================================================
# PDR evaluation
# ============================================================

def run_episode_pdr(
    env,
    rule: str,
):
    env.reset()

    done = False
    last_info = None

    start_time = time.perf_counter()

    while not done:

        valid_actions = env.get_valid_actions()

        if not valid_actions:
            env._advance_to_next_event()
            valid_actions = env.get_valid_actions()

        action = select_pdr_action(
            env,
            rule,
        )

        _, _, done, info = env.step(action)

        last_info = info

    solve_time = (
        time.perf_counter()
        - start_time
    )

    return (
        float(last_info["makespan"]),
        float(solve_time),
    )


# ============================================================
# Checkpoint loader
# ============================================================

def load_actor_checkpoint(
    actor,
    checkpoint_path,
    map_location="cpu",
):
    """
    Support two checkpoint formats:

    1. pure actor state_dict
    2. full training checkpoint containing actor_state_dict
    """

    ckpt = torch.load(
        checkpoint_path,
        map_location=map_location,
    )

    if (
        isinstance(ckpt, dict)
        and "actor_state_dict" in ckpt
    ):
        actor.load_state_dict(
            ckpt["actor_state_dict"]
        )

    else:
        actor.load_state_dict(ckpt)


# ============================================================
# Evaluate RL method
# ============================================================

def evaluate_rl_method(
    actor_ckpt_path,
    instances,
    seeds,
    use_fatigue,
    hidden_dim=64,
    num_layers=2,
):
    actor = SHyperActorFull(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    q1 = SHyperQCriticFull(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    q2 = SHyperQCriticFull(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    agent = SACAgent(
        actor=actor,
        q1=q1,
        q2=q2,
        lr=1e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.1,
        device="cpu",
    )

    load_actor_checkpoint(
        actor,
        actor_ckpt_path,
        map_location="cpu",
    )

    actor.eval()

    makespans = []
    solve_times = []

    for seed, instance in zip(
        seeds,
        instances,
    ):

        env = make_env(
            instance=instance,
            use_fatigue=use_fatigue,
        )

        makespan, solve_time = run_episode(
            env,
            agent,
        )

        makespans.append(
            makespan
        )

        solve_times.append(
            solve_time
        )

        print(
            f"    seed={seed} | "
            f"makespan={makespan:.2f} | "
            f"time={solve_time:.4f}s"
        )

    avg_makespan = (
        sum(makespans)
        / len(makespans)
    )

    worst_makespan = max(
        makespans
    )

    avg_solve_time = (
        sum(solve_times)
        / len(solve_times)
    )

    return {
        "avg_makespan":
            round(float(avg_makespan), 2),

        "worst_makespan":
            round(float(worst_makespan), 2),

        "avg_solve_time_seconds":
            round(float(avg_solve_time), 4),

        "per_seed":
            [
                round(float(x), 2)
                for x in makespans
            ],
    }


# ============================================================
# Evaluate PDR method
# ============================================================

def evaluate_pdr_method(
    rule,
    instances,
    seeds,
    use_fatigue,
):
    makespans = []
    solve_times = []

    for seed, instance in zip(
        seeds,
        instances,
    ):

        env = make_env(
            instance=instance,
            use_fatigue=use_fatigue,
        )

        makespan, solve_time = run_episode_pdr(
            env,
            rule,
        )

        makespans.append(
            makespan
        )

        solve_times.append(
            solve_time
        )

        print(
            f"    seed={seed} | "
            f"makespan={makespan:.2f} | "
            f"time={solve_time:.4f}s"
        )

    avg_makespan = (
        sum(makespans)
        / len(makespans)
    )

    worst_makespan = max(
        makespans
    )

    avg_solve_time = (
        sum(solve_times)
        / len(solve_times)
    )

    return {
        "avg_makespan":
            round(float(avg_makespan), 2),

        "worst_makespan":
            round(float(worst_makespan), 2),

        "avg_solve_time_seconds":
            round(float(avg_solve_time), 4),

        "per_seed":
            [
                round(float(x), 2)
                for x in makespans
            ],
    }


# ============================================================
# Evaluate CP-SAT exact method
# ============================================================

def evaluate_exact_method(
    instances,
    seeds,
    exact_fatigue_level,
    time_limit_seconds=3600.0,
):
    """
    CP-SAT exact evaluation.

    exact_fatigue_level = 0.0:
        F = 0
        p_ijkq = p_ijk

    exact_fatigue_level = 1.0:
        F = 1
        p_ijkq = (1 + ln 2) * p_ijk

    CP-SAT itself solves the F=0 problem.

    Since fixed F=1 applies the same multiplicative factor
    to every processing time, the optimal schedule is unchanged
    and the optimal makespan is multiplied by 1 + ln(2).
    """

    if exact_fatigue_level not in (
        0.0,
        1.0,
    ):
        raise ValueError(
            "exact_fatigue_level must be "
            "either 0.0 or 1.0."
        )

    fatigue_factor = (
        1.0
        + math.log(
            1.0
            + exact_fatigue_level
        )
    )

    makespans = []
    base_makespans = []
    best_bounds = []
    solve_times = []
    statuses = []

    for seed, instance in zip(
        seeds,
        instances,
    ):

        print(
            f"    CP-SAT seed={seed} | "
            f"fixed F={exact_fatigue_level}"
        )

        solver = CPSATSolver(
            instance=instance,
            time_limit=time_limit_seconds,
            time_scale=1,
        )

        result = solver.solve()

        status = result["status"]

        objective = result["objective"]

        best_bound = result["best_bound"]

        if objective is None:
            raise RuntimeError(
                f"CP-SAT failed on seed={seed}. "
                f"status={status}"
            )

        # ----------------------------------------------------
        # For an exact solution comparison, optimality must
        # actually be proven.
        # ----------------------------------------------------
        if not result["is_optimal"]:

            raise RuntimeError(
                f"CP-SAT did not prove optimality "
                f"for seed={seed}. "
                f"status={status}, "
                f"objective={objective}, "
                f"best_bound={best_bound}"
            )

        base_makespan = float(
            objective
        )

        fixed_fatigue_makespan = (
            base_makespan
            * fatigue_factor
        )

        fixed_fatigue_bound = (
            float(best_bound)
            * fatigue_factor
        )

        makespans.append(
            fixed_fatigue_makespan
        )

        base_makespans.append(
            base_makespan
        )

        best_bounds.append(
            fixed_fatigue_bound
        )

        solve_times.append(
            float(result["wall_time"])
        )

        statuses.append(
            status
        )

        print(
            f"        status={status} | "
            f"F=0 optimum={base_makespan:.2f} | "
            f"reported makespan="
            f"{fixed_fatigue_makespan:.2f} | "
            f"time={result['wall_time']:.2f}s"
        )

    avg_makespan = (
        sum(makespans)
        / len(makespans)
    )

    worst_makespan = max(
        makespans
    )

    avg_solve_time = (
        sum(solve_times)
        / len(solve_times)
    )

    return {
        "avg_makespan":
            round(float(avg_makespan), 2),

        "worst_makespan":
            round(float(worst_makespan), 2),

        "avg_solve_time_seconds":
            round(float(avg_solve_time), 4),

        "per_seed":
            [
                round(float(x), 2)
                for x in makespans
            ],

        "per_seed_f0_optimum":
            [
                round(float(x), 2)
                for x in base_makespans
            ],

        "per_seed_best_bound":
            [
                round(float(x), 2)
                for x in best_bounds
            ],

        "per_seed_status":
            statuses,

        "fixed_fatigue_level":
            float(exact_fatigue_level),

        "fatigue_factor":
            float(fatigue_factor),

        "all_optimal":
            True,
    }


# ============================================================
# GA evaluation
#
# Keep the current external-result design for now.
# We will modify this in the next step.
# ============================================================

def evaluate_ga_method(
    ga_json_path,
    expected_seeds,
):
    with open(
        ga_json_path,
        "r",
        encoding="utf-8",
    ) as f:
        ga_summary = json.load(f)

    ga_results = ga_summary[
        "results"
    ]

    ga_seed_to_result = {
        int(item["seed"]):
            float(item["best_makespan"])
        for item in ga_results
    }

    makespans = []
    missing = []

    for seed in expected_seeds:

        if seed not in ga_seed_to_result:
            missing.append(seed)

        else:
            makespans.append(
                ga_seed_to_result[seed]
            )

    if missing:
        raise ValueError(
            f"GA results missing seeds: "
            f"{missing}"
        )

    avg_makespan = (
        sum(makespans)
        / len(makespans)
    )

    worst_makespan = max(
        makespans
    )

    return {
        "avg_makespan":
            round(float(avg_makespan), 2),

        "worst_makespan":
            round(float(worst_makespan), 2),

        "per_seed":
            [
                round(float(x), 2)
                for x in makespans
            ],

        "source_json":
            ga_json_path,
    }


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # UNIFIED EVALUATION CONFIGURATION
    # ========================================================

    eval_config = {

        # ----------------------------------------------------
        # Instance setting
        # ----------------------------------------------------

        "seeds": list(
            range(500, 600)
        ),

        "num_jobs": 3,

        "num_machines": 3,

        "num_workers": 2,

        "min_ops_per_job": 3,

        "max_ops_per_job": 7,

        # ----------------------------------------------------
        # Network setting
        # ----------------------------------------------------

        "hidden_dim": 64,

        "num_layers": 2,

        # ====================================================
        # EXPERIMENT CONTROL
        # ====================================================

        # 1.
        # Whether CP-SAT participates in this experiment.
        #
        # True  -> evaluate CP-SAT
        # False -> do not evaluate CP-SAT
        "include_exact": True,

        # 2.
        # Whether NON-EXACT methods use PGNN fatigue.
        #
        # This controls:
        # RL methods
        # PDR methods
        #
        # True:
        #     dynamic PGNN fatigue is enabled
        #
        # False:
        #     fatigue is disabled
        #     F = 0
        #
        # This parameter does NOT control CP-SAT.
        "use_fatigue": True,

        # 3.
        # Fixed fatigue level used by the exact method.
        #
        # Allowed:
        #     0.0 -> F = 0
        #     1.0 -> F = 1
        #
        # Only active when include_exact = True.
        "exact_fatigue_level": 0.0,

        # Maximum CP-SAT time for ONE instance.
        "exact_time_limit_seconds": 3600.0,
    }

    # ========================================================
    # Generate all instances ONCE
    # ========================================================

    instances = build_instances(
        seeds=eval_config["seeds"],
        num_jobs=eval_config[
            "num_jobs"
        ],
        num_machines=eval_config[
            "num_machines"
        ],
        num_workers=eval_config[
            "num_workers"
        ],
        min_ops_per_job=eval_config[
            "min_ops_per_job"
        ],
        max_ops_per_job=eval_config[
            "max_ops_per_job"
        ],
    )

    # ========================================================
    # Print experiment configuration
    # ========================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "Evaluation configuration"
    )

    print(
        "============================================================"
    )

    print(
        f"Instances          : "
        f"{len(instances)}"
    )

    print(
        f"Seeds              : "
        f"{eval_config['seeds']}"
    )

    print(
        f"Problem scale      : "
        f"{eval_config['num_jobs']} jobs x "
        f"{eval_config['num_machines']} machines x "
        f"{eval_config['num_workers']} workers"
    )

    print(
        f"Operations/job     : "
        f"{eval_config['min_ops_per_job']}"
        f"-"
        f"{eval_config['max_ops_per_job']}"
    )

    print(
        f"Include exact      : "
        f"{eval_config['include_exact']}"
    )

    print(
        f"Use fatigue        : "
        f"{eval_config['use_fatigue']} "
        f"(non-exact methods)"
    )

    if eval_config[
        "include_exact"
    ]:
        print(
            f"Exact fatigue      : "
            f"F="
            f"{eval_config['exact_fatigue_level']}"
        )

        print(
            f"Exact time limit   : "
            f"{eval_config['exact_time_limit_seconds']}"
            f"s / instance"
        )

    print(
        "============================================================"
    )

    # ========================================================
    # Methods
    # ========================================================

    methods = {

        # ----------------------------------------------------
        # SAC
        # ----------------------------------------------------

        "full_sac": {
            "type": "rl",

            "actor_ckpt":
                "best_fixedscale_full_sac_actor.pt",
        },

        # ----------------------------------------------------
        # Preference RL
        # ----------------------------------------------------

        "pref_3stage_parallel_best": {
            "type": "rl",

            "actor_ckpt":
                "best_pref3stage_parallel_actor.pt",
        },

        # ----------------------------------------------------
        # Archived preference checkpoints
        # ----------------------------------------------------

        "pref_step3_iter_40": {
            "type": "rl",

            "actor_ckpt":
                "checkpoints/archive/"
                "ckpt_step3_iter_0040.pt",
        },

        "pref_step3_iter_0250": {
            "type": "rl",

            "actor_ckpt":
                "checkpoints/archive/"
                "ckpt_step3_iter_0250.pt",
        },

        # ----------------------------------------------------
        # Dispatching rules
        # ----------------------------------------------------

        "fifo": {
            "type": "pdr",
            "rule": "FIFO",
        },

        "spt": {
            "type": "pdr",
            "rule": "SPT",
        },

        "mwkr": {
            "type": "pdr",
            "rule": "MWKR",
        },

        # ----------------------------------------------------
        # GA
        #
        # Keep external JSON temporarily.
        # We will integrate GA directly in the next step.
        # ----------------------------------------------------

        # "ga_baseline": {
        #     "type": "ga",
        #     "json_path":
        #         "eval_results/"
        #         "ga_baseline_summary.json",
        # },
    }

    # ========================================================
    # Add exact method if requested
    # ========================================================

    if eval_config[
        "include_exact"
    ]:

        methods = {
            "cp_sat_exact": {
                "type": "exact",
            },

            **methods,
        }

    # ========================================================
    # Run evaluation
    # ========================================================

    results = {}

    for (
        method_name,
        method_cfg,
    ) in methods.items():

        print(
            f"\n"
            f"============================================================"
        )

        print(
            f"Evaluating method: "
            f"{method_name}"
        )

        print(
            f"============================================================"
        )

        # ----------------------------------------------------
        # RL
        # ----------------------------------------------------

        if method_cfg[
            "type"
        ] == "rl":

            ckpt_path = method_cfg[
                "actor_ckpt"
            ]

            if not os.path.exists(
                ckpt_path
            ):
                print(
                    f"Checkpoint not found: "
                    f"{ckpt_path}, skip."
                )
                continue

            result = evaluate_rl_method(
                actor_ckpt_path=
                    ckpt_path,

                instances=
                    instances,

                seeds=
                    eval_config[
                        "seeds"
                    ],

                use_fatigue=
                    eval_config[
                        "use_fatigue"
                    ],

                hidden_dim=
                    eval_config[
                        "hidden_dim"
                    ],

                num_layers=
                    eval_config[
                        "num_layers"
                    ],
            )

        # ----------------------------------------------------
        # PDR
        # ----------------------------------------------------

        elif method_cfg[
            "type"
        ] == "pdr":

            result = evaluate_pdr_method(
                rule=
                    method_cfg[
                        "rule"
                    ],

                instances=
                    instances,

                seeds=
                    eval_config[
                        "seeds"
                    ],

                use_fatigue=
                    eval_config[
                        "use_fatigue"
                    ],
            )

        # ----------------------------------------------------
        # Exact CP-SAT
        # ----------------------------------------------------

        elif method_cfg[
            "type"
        ] == "exact":

            result = evaluate_exact_method(
                instances=
                    instances,

                seeds=
                    eval_config[
                        "seeds"
                    ],

                exact_fatigue_level=
                    eval_config[
                        "exact_fatigue_level"
                    ],

                time_limit_seconds=
                    eval_config[
                        "exact_time_limit_seconds"
                    ],
            )

        # ----------------------------------------------------
        # GA
        # ----------------------------------------------------

        elif method_cfg[
            "type"
        ] == "ga":

            json_path = method_cfg[
                "json_path"
            ]

            if not os.path.exists(
                json_path
            ):
                print(
                    f"GA result json not found: "
                    f"{json_path}, skip."
                )
                continue

            result = evaluate_ga_method(
                ga_json_path=
                    json_path,

                expected_seeds=
                    eval_config[
                        "seeds"
                    ],
            )

        else:
            raise ValueError(
                f"Unknown method type: "
                f"{method_cfg['type']}"
            )

        results[
            method_name
        ] = result

        # ----------------------------------------------------
        # Print method summary
        # ----------------------------------------------------

        if (
            "avg_solve_time_seconds"
            in result
        ):

            print(
                f"\n"
                f"{method_name} | "
                f"avg_makespan="
                f"{result['avg_makespan']:.2f} | "
                f"worst_makespan="
                f"{result['worst_makespan']:.2f} | "
                f"avg_solve_time="
                f"{result['avg_solve_time_seconds']:.4f}s"
            )

        else:

            print(
                f"\n"
                f"{method_name} | "
                f"avg_makespan="
                f"{result['avg_makespan']:.2f} | "
                f"worst_makespan="
                f"{result['worst_makespan']:.2f}"
            )

        print(
            "per_seed:",
            result["per_seed"],
        )

    # ========================================================
    # Relative gap to best obtained average result
    #
    # This retains the original evaluation logic.
    # It is NOT necessarily an optimality gap.
    # ========================================================

    valid_avg_makespans = [
        res["avg_makespan"]
        for res in results.values()
        if "avg_makespan" in res
    ]

    if not valid_avg_makespans:
        raise RuntimeError(
            "No valid evaluation results."
        )

    best_makespan = min(
        valid_avg_makespans
    )

    for (
        method_name,
        res,
    ) in results.items():

        gap = (
            (
                res["avg_makespan"]
                - best_makespan
            )
            / best_makespan
            * 100.0
        )

        res[
            "gap_percent"
        ] = round(
            float(gap),
            2,
        )

    # ========================================================
    # TRUE OPTIMALITY GAP
    #
    # Only valid when:
    #
    # 1. CP-SAT is included
    # 2. non-exact methods use F = 0
    # 3. exact method also uses F = 0
    #
    # Then all methods solve the SAME problem.
    # ========================================================

    exact_comparable = (
        eval_config[
            "include_exact"
        ]
        and
        (
            not eval_config[
                "use_fatigue"
            ]
        )
        and
        abs(
            eval_config[
                "exact_fatigue_level"
            ]
            - 0.0
        ) < 1e-12
        and
        "cp_sat_exact"
        in results
    )

    eval_config[
        "exact_comparable"
    ] = bool(
        exact_comparable
    )

    if exact_comparable:

        print(
            "\n"
            "============================================================"
        )

        print(
            "Computing exact optimality gaps"
        )

        print(
            "============================================================"
        )

        exact_per_seed = results[
            "cp_sat_exact"
        ][
            "per_seed"
        ]

        for (
            method_name,
            res,
        ) in results.items():

            method_per_seed = res[
                "per_seed"
            ]

            if (
                len(method_per_seed)
                != len(exact_per_seed)
            ):
                raise ValueError(
                    f"Seed count mismatch "
                    f"for method "
                    f"{method_name}."
                )

            per_seed_gaps = []

            for (
                method_value,
                exact_value,
            ) in zip(
                method_per_seed,
                exact_per_seed,
            ):

                if exact_value <= 0:
                    raise ValueError(
                        "Exact makespan must "
                        "be positive."
                    )

                gap = (
                    (
                        method_value
                        - exact_value
                    )
                    / exact_value
                    * 100.0
                )

                per_seed_gaps.append(
                    gap
                )

            res[
                "per_seed_optimality_gap_percent"
            ] = [
                round(
                    float(x),
                    2,
                )
                for x
                in per_seed_gaps
            ]

            res[
                "optimality_gap_percent"
            ] = round(
                float(
                    sum(per_seed_gaps)
                    / len(per_seed_gaps)
                ),
                2,
            )

            print(
                f"{method_name} | "
                f"average optimality gap="
                f"{res['optimality_gap_percent']:.2f}%"
            )

    else:

        print(
            "\nExact optimality gap is NOT calculated."
        )

        print(
            "Reason: CP-SAT and the other methods "
            "are not solving the same F=0 problem."
        )

    # ========================================================
    # Ranking
    # ========================================================

    ranked = sorted(
        results.items(),
        key=lambda kv:
            kv[1][
                "avg_makespan"
            ],
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "Ranking by avg_makespan"
    )

    print(
        "============================================================"
    )

    for (
        rank,
        (
            method_name,
            result,
        ),
    ) in enumerate(
        ranked,
        start=1,
    ):

        line = (
            f"{rank}. "
            f"{method_name} | "
            f"avg="
            f"{result['avg_makespan']:.2f} | "
            f"worst="
            f"{result['worst_makespan']:.2f}"
        )

        if (
            "avg_solve_time_seconds"
            in result
        ):
            line += (
                f" | time="
                f"{result['avg_solve_time_seconds']:.4f}s"
            )

        line += (
            f" | relative_gap="
            f"{result['gap_percent']:.2f}%"
        )

        if (
            "optimality_gap_percent"
            in result
        ):
            line += (
                f" | optimality_gap="
                f"{result['optimality_gap_percent']:.2f}%"
            )

        print(line)

    # ========================================================
    # Save results
    # ========================================================

    os.makedirs(
        "eval_results",
        exist_ok=True,
    )

    output = {

        "eval_config":
            eval_config,

        "best_obtained_makespan":
            round(
                float(
                    best_makespan
                ),
                2,
            ),

        "results":
            results,

        "ranking": [],
    }

    for (
        rank,
        (
            method_name,
            result,
        ),
    ) in enumerate(
        ranked,
        start=1,
    ):

        ranking_item = {

            "rank":
                rank,

            "method":
                method_name,

            "avg_makespan":
                result[
                    "avg_makespan"
                ],

            "worst_makespan":
                result[
                    "worst_makespan"
                ],

            "gap_percent":
                result[
                    "gap_percent"
                ],
        }

        if (
            "avg_solve_time_seconds"
            in result
        ):
            ranking_item[
                "avg_solve_time_seconds"
            ] = result[
                "avg_solve_time_seconds"
            ]

        if (
            "optimality_gap_percent"
            in result
        ):
            ranking_item[
                "optimality_gap_percent"
            ] = result[
                "optimality_gap_percent"
            ]

        output[
            "ranking"
        ].append(
            ranking_item
        )

    output_path = (
        "eval_results/"
        "evaluation_all_methods.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nSaved to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()