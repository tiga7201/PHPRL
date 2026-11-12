import argparse
import json
import os
from datetime import datetime

from env.instance_generator import generate_random_instance
from exact.scip_fatigue_solver import SCIPFatigueSolver


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the exact exponential-fatigue MINLP with SCIP."
    )
    parser.add_argument("--seed-start", type=int, default=500)
    parser.add_argument("--num-instances", type=int, default=100)
    parser.add_argument("--num-jobs", type=int, default=3)
    parser.add_argument("--num-machines", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-ops-per-job", type=int, default=3)
    parser.add_argument("--max-ops-per-job", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--relative-gap", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--log-output", action="store_true")
    parser.add_argument("--output-dir", default="eval_results")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.num_instances))
    records = []

    for index, seed in enumerate(seeds, start=1):
        print(
            f"[{index}/{len(seeds)}] SCIP exponential fatigue | seed={seed} | "
            f"limit={args.time_limit:.0f}s"
        )
        instance = generate_random_instance(
            seed=seed,
            num_jobs=args.num_jobs,
            num_machines=args.num_machines,
            num_workers=args.num_workers,
            min_ops_per_job=args.min_ops_per_job,
            max_ops_per_job=args.max_ops_per_job,
        )
        solver = SCIPFatigueSolver(
            instance=instance,
            time_limit=args.time_limit,
            num_threads=args.num_threads,
            relative_gap=args.relative_gap,
            log_output=args.log_output,
        )
        result = solver.solve()
        gap_percent = (
            None
            if result["relative_gap"] is None
            else float(result["relative_gap"]) * 100.0
        )
        record = {
            "seed": seed,
            "status": result["status"],
            "is_optimal": bool(result["is_optimal"]),
            "has_solution": bool(result["has_solution"]),
            "makespan": result["objective"],
            "best_bound": result["best_bound"],
            "solver_gap_percent": gap_percent,
            "solve_time_seconds": result["wall_time"],
            "num_solutions": result["num_solutions"],
            "num_nodes": result["num_nodes"],
            "schedule": result["schedule"],
        }
        records.append(record)

        if result["has_solution"]:
            label = "optimal" if result["is_optimal"] else "best feasible"
            print(
                f"    {label}: makespan={result['objective']:.4f} | "
                f"bound={result['best_bound']} | gap={gap_percent} | "
                f"time={result['wall_time']:.2f}s"
            )
        else:
            print(f"    no feasible solution | status={result['status']}")

    feasible_records = [record for record in records if record["has_solution"]]
    optimal_records = [record for record in records if record["is_optimal"]]
    makespans = [float(record["makespan"]) for record in feasible_records]
    solve_times = [float(record["solve_time_seconds"]) for record in records]

    summary = {
        "method": "scip_exponential_fatigue_minlp",
        "fatigue_model": {
            "processing_time": "p * (1 + ln(1 + F_start))",
            "working": "1 - (1 - F_start) * exp(-lambda * processing_time)",
            "resting": "F_previous_end * exp(-mu * rest_time)",
            "physical_condition_rates": {
                "1": {"lambda": 0.05, "mu": 0.01},
                "2": {"lambda": 0.03, "mu": 0.03},
                "3": {"lambda": 0.01, "mu": 0.05},
            },
            "initial_fatigue": 0.0,
            "skill_level": 1.0,
        },
        "instance_config": {
            "num_jobs": args.num_jobs,
            "num_machines": args.num_machines,
            "num_workers": args.num_workers,
            "min_ops_per_job": args.min_ops_per_job,
            "max_ops_per_job": args.max_ops_per_job,
            "seeds": seeds,
        },
        "solver_config": {
            "time_limit_seconds_per_instance": args.time_limit,
            "relative_gap_limit": args.relative_gap,
            "num_threads": args.num_threads,
        },
        "num_instances": len(records),
        "num_feasible": len(feasible_records),
        "num_optimal": len(optimal_records),
        "num_feasible_not_optimal": len(feasible_records) - len(optimal_records),
        "num_without_solution": len(records) - len(feasible_records),
        "optimal_rate_percent": (
            100.0 * len(optimal_records) / len(records) if records else 0.0
        ),
        "avg_makespan": sum(makespans) / len(makespans) if makespans else None,
        "worst_makespan": max(makespans) if makespans else None,
        "avg_solve_time_seconds": (
            sum(solve_times) / len(solve_times) if solve_times else None
        ),
        "per_seed_makespan": [record["makespan"] for record in records],
        "per_seed_best_bound": [record["best_bound"] for record in records],
        "per_seed_status": [record["status"] for record in records],
        "per_seed_is_optimal": [record["is_optimal"] for record in records],
        "per_seed_solver_gap_percent": [
            record["solver_gap_percent"] for record in records
        ],
        "records": records,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    size_suffix = (
        f"{args.num_jobs}j_{args.num_machines}m_{args.num_workers}w_"
        f"{args.min_ops_per_job}-{args.max_ops_per_job}ops"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir,
        f"scip_fatigue_{size_suffix}_{timestamp}.json",
    )
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\nSCIP fatigue evaluation summary")
    print(f"    feasible={len(feasible_records)}/{len(records)}")
    print(f"    optimal={len(optimal_records)}/{len(records)}")
    print(f"    avg_makespan={summary['avg_makespan']}")
    print(f"    avg_time={summary['avg_solve_time_seconds']}")
    print(f"    saved={output_path}")


if __name__ == "__main__":
    main()
