import json
import os

from rl.ga_baseline import evaluate_ga_on_seeds


def main():
    config = {
        "seeds": [100, 101, 102, 103, 104],
        "num_jobs": 40,
        "num_machines": 5,
        "num_workers": 3,
        "min_ops_per_job": 2,
        "max_ops_per_job": 4,
        "pop_size": 30,
        "generations": 50,
        "mut_rate": 0.1,
        "elite_size": 2,
    }

    results = evaluate_ga_on_seeds(
        seeds=config["seeds"],
        num_jobs=config["num_jobs"],
        num_machines=config["num_machines"],
        num_workers=config["num_workers"],
        min_ops_per_job=config["min_ops_per_job"],
        max_ops_per_job=config["max_ops_per_job"],
        pop_size=config["pop_size"],
        generations=config["generations"],
        mut_rate=config["mut_rate"],
        elite_size=config["elite_size"],
    )

    avg_makespan = sum(x["best_makespan"] for x in results) / len(results)
    worst_makespan = max(x["best_makespan"] for x in results)

    summary = {
        "config": config,
        "avg_makespan": avg_makespan,
        "worst_makespan": worst_makespan,
        "results": results,
    }

    os.makedirs("eval_results", exist_ok=True)
    with open("eval_results/ga_baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nGA baseline finished.")
    print(f"Average makespan: {avg_makespan:.4f}")
    print(f"Worst makespan: {worst_makespan:.4f}")


if __name__ == "__main__":
    main()