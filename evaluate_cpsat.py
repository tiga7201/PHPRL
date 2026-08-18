import json
import os

from env.instance_generator import load_instance_dataset
from exact.cpsat_solver import CPSATSolver


BENCHMARK_PATH = "benchmark_exact_3x3x3.json"
OUTPUT_PATH = "eval_results/cpsat_exact_results.json"

TIME_LIMIT = 3600.0
TIME_SCALE = 1


def main():

    instances = load_instance_dataset(BENCHMARK_PATH)

    print(f"Loaded {len(instances)} instances from {BENCHMARK_PATH}")

    results = []

    for idx, instance in enumerate(instances):

        print(
            f"\n========== Instance {idx + 1}/{len(instances)} =========="
        )

        solver = CPSATSolver(
            instance=instance,
            time_limit=TIME_LIMIT,
            time_scale=TIME_SCALE,
        )

        result = solver.solve()

        result["instance_index"] = idx
        result["num_jobs"] = instance.num_jobs
        result["num_machines"] = instance.num_machines
        result["num_workers"] = instance.num_workers
        result["num_operations"] = sum(
            len(job) for job in instance.jobs
        )

        results.append(result)

        print(f"Status       : {result['status']}")
        print(f"Wall time    : {result['wall_time']:.4f} s")

        if result["objective"] is not None:
            print(f"Objective    : {result['objective']:.4f}")
            print(f"Best bound   : {result['best_bound']:.4f}")

        if result["is_optimal"]:
            print("Optimality   : PROVEN")
        else:
            print("Optimality   : NOT PROVEN")

    optimal_results = [
        x for x in results
        if x["is_optimal"]
    ]

    summary = {
        "benchmark_path": BENCHMARK_PATH,
        "num_instances": len(results),
        "num_optimal": len(optimal_results),
        "optimal_ratio": (
            len(optimal_results) / len(results)
            if results else 0.0
        ),
        "average_wall_time": (
            sum(x["wall_time"] for x in results) / len(results)
            if results else 0.0
        ),
    }

    if optimal_results:
        summary["average_optimal_makespan"] = (
            sum(x["objective"] for x in optimal_results)
            / len(optimal_results)
        )

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    payload = {
        "summary": summary,
        "results": results,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n========== Summary ==========")
    print(
        f"Optimal instances : "
        f"{summary['num_optimal']}/{summary['num_instances']}"
    )
    print(
        f"Optimal ratio     : "
        f"{summary['optimal_ratio'] * 100:.2f}%"
    )
    print(
        f"Average wall time : "
        f"{summary['average_wall_time']:.4f} s"
    )
    print(f"Saved to          : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()