import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from env.instance_generator import generate_random_instance

from baseline.hga.decoder import HgaDecoder
from baseline.hga.hga import HGA, HGAConfig
from baseline.hga.local_search import (
    CriticalPathLocalSearch,
    LocalSearchConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HGA with the complete Phase-1+2 PGNN."
    )

    parser.add_argument("--seed-start", type=int, default=500)
    parser.add_argument("--seed-end", type=int, default=501)

    parser.add_argument("--num-jobs", type=int, default=100)
    parser.add_argument("--num-machines", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--min-ops-per-job", type=int, default=3)
    parser.add_argument("--max-ops-per-job", type=int, default=7)

    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--crossover-rate", type=float, default=0.9)

    parser.add_argument(
        "--os-mutation-rate",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--machine-mutation-rate",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--worker-mutation-rate",
        type=float,
        default=0.1,
    )

    parser.add_argument("--elite-size", type=int, default=2)
    parser.add_argument("--tournament-size", type=int, default=3)

    parser.add_argument(
        "--local-search-elite-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--local-search-interval",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--local-search-iterations",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--local-search-neighbors",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-time-per-instance",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--stagnation-generations",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--phase1-checkpoint",
        type=str,
        default="checkpoints/pgnn_phase1.pt",
    )
    parser.add_argument(
        "--phase2-checkpoint",
        type=str,
        default="checkpoints/pgnn_phase2.pt",
    )

    parser.add_argument(
        "--algorithm-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output",
        type=str,
        # default="eval_results/hga_pgnn12_summary.json",
        default=None,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--verbose-hga",
        action="store_true",
    )

    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def build_summary(
    config: dict,
    expected_seeds: List[int],
    results: List[dict],
) -> dict:
    ordered_results = sorted(
        results,
        key=lambda item: int(item["seed"]),
    )

    result_by_seed = {
        int(item["seed"]): item
        for item in ordered_results
    }

    # Makespans are aligned with seed_order. In incomplete runs,
    # unfinished seeds are represented by null in JSON.
    per_seed_makespan = [
        (
            float(
                result_by_seed[seed]["best_makespan"]
            )
            if seed in result_by_seed
            else None
        )
        for seed in expected_seeds
    ]

    makespans = [
        value
        for value in per_seed_makespan
        if value is not None
    ]

    solve_times = [
        float(item["solve_time_seconds"])
        for item in ordered_results
    ]

    missing_seeds = [
        seed
        for seed in expected_seeds
        if seed not in result_by_seed
    ]

    return {
        "method": "critical_path_hga",
        "fatigue_model": "pgnn_phase1_phase2",
        "status": (
            "complete"
            if not missing_seeds
            else "partial"
        ),
        "config": config,
        "num_completed": len(ordered_results),
        "num_expected": len(expected_seeds),
        "missing_seeds": missing_seeds,

        # These two lists use the same positions.
        "seed_order": expected_seeds,
        "per_seed_makespan": per_seed_makespan,

        "avg_makespan": (
            sum(makespans) / len(makespans)
            if makespans
            else None
        ),
        "worst_makespan": (
            max(makespans)
            if makespans
            else None
        ),
        "best_makespan": (
            min(makespans)
            if makespans
            else None
        ),
        "avg_solve_time_seconds": (
            sum(solve_times) / len(solve_times)
            if solve_times
            else None
        ),
        "results": ordered_results,
    }

def load_existing_results(
    output_path: Path,
    expected_config: dict,
    resume: bool,
) -> Dict[int, dict]:
    if not resume or not output_path.is_file():
        return {}

    existing = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    existing_config = existing.get("config")

    if existing_config != expected_config:
        raise ValueError(
            "Existing result configuration differs from the "
            "current configuration. Use another output path or "
            "run without --resume."
        )

    return {
        int(item["seed"]): item
        for item in existing.get("results", [])
    }


def main():
    args = parse_args()

    phase1_path = Path(args.phase1_checkpoint)
    phase2_path = Path(args.phase2_checkpoint)
    if args.output is None:
        output_filename = (
            f"hga_pgnn12_"
            f"J{args.num_jobs}_"
            f"M{args.num_machines}_"
            f"W{args.num_workers}_"
            f"O{args.min_ops_per_job}-{args.max_ops_per_job}_"
            f"S{args.seed_start}-{args.seed_end - 1}.json"
        )
        output_path = (
                Path("eval_results")
                / output_filename
        )
    else:
        output_path = Path(args.output)

    if not phase1_path.is_file():
        raise FileNotFoundError(
            f"Phase-1 checkpoint not found: {phase1_path}"
        )

    if not phase2_path.is_file():
        raise FileNotFoundError(
            f"Phase-2 checkpoint not found: {phase2_path}"
        )

    seeds = list(
        range(args.seed_start, args.seed_end)
    )

    if not seeds:
        raise ValueError(
            "The selected seed range is empty."
        )

    hga_config = HGAConfig(
        population_size=args.population_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        os_mutation_rate=args.os_mutation_rate,
        machine_mutation_rate=(
            args.machine_mutation_rate
        ),
        worker_mutation_rate=(
            args.worker_mutation_rate
        ),
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        local_search_elite_size=(
            args.local_search_elite_size
        ),
        local_search_interval=(
            args.local_search_interval
        ),
        max_time_seconds=(
            args.max_time_per_instance
        ),
        stagnation_generations=(
            args.stagnation_generations
        ),
        verbose=args.verbose_hga,
    )

    local_search_config = LocalSearchConfig(
        max_iterations=(
            args.local_search_iterations
        ),
        max_neighbors=(
            args.local_search_neighbors
        ),
        strategy="best",
    )

    experiment_config = {
        "seeds": seeds,
        "instance": {
            "num_jobs": args.num_jobs,
            "num_machines": args.num_machines,
            "num_workers": args.num_workers,
            "min_ops_per_job": args.min_ops_per_job,
            "max_ops_per_job": args.max_ops_per_job,
        },
        "hga": asdict(hga_config),
        "local_search": asdict(
            local_search_config
        ),
        "algorithm_seed": args.algorithm_seed,
        "phase1_checkpoint": str(phase1_path),
        "phase2_checkpoint": str(phase2_path),
    }

    results_by_seed = load_existing_results(
        output_path=output_path,
        expected_config=experiment_config,
        resume=args.resume,
    )

    for position, instance_seed in enumerate(
        seeds,
        start=1,
    ):
        if instance_seed in results_by_seed:
            print(
                f"[{position}/{len(seeds)}] "
                f"seed={instance_seed} already completed."
            )
            continue

        print(
            f"[{position}/{len(seeds)}] "
            f"seed={instance_seed} started."
        )

        instance = generate_random_instance(
            seed=instance_seed,
            num_jobs=args.num_jobs,
            num_machines=args.num_machines,
            num_workers=args.num_workers,
            min_ops_per_job=args.min_ops_per_job,
            max_ops_per_job=args.max_ops_per_job,
        )

        decoder = HgaDecoder(
            instance=instance,
            phase1_checkpoint=str(phase1_path),
            phase2_checkpoint=str(phase2_path),
            device="cpu",
            use_fatigue=True,
        )

        search_seed = (
            args.algorithm_seed + instance_seed
        )

        local_search = CriticalPathLocalSearch(
            instance=instance,
            decoder=decoder,
            config=local_search_config,
            seed=search_seed + 100_000,
        )

        solver = HGA(
            instance=instance,
            decoder=decoder,
            config=hga_config,
            local_search=local_search,
            seed=search_seed,
        )

        result = solver.solve()

        record = {
            "seed": instance_seed,
            "best_makespan": (
                result.best_makespan
            ),
            "solve_time_seconds": (
                result.elapsed_seconds
            ),
            "generations_completed": (
                result.generations_completed
            ),
            "evaluations": result.evaluations,
            "ga_evaluations": (
                result.ga_evaluations
            ),
            "local_search_evaluations": (
                result.local_search_evaluations
            ),
            "best_chromosome": {
                "OS": result.best_chromosome.OS,
                "MS": result.best_chromosome.MS,
                "WS": result.best_chromosome.WS,
            },
            "history": result.history,
        }

        results_by_seed[instance_seed] = record

        summary = build_summary(
            config=experiment_config,
            expected_seeds=seeds,
            results=list(
                results_by_seed.values()
            ),
        )
        atomic_write_json(output_path, summary)

        print(
            f"seed={instance_seed} finished | "
            f"makespan={result.best_makespan:.4f} | "
            f"time={result.elapsed_seconds:.2f}s | "
            f"evaluations={result.evaluations}"
        )

    final_summary = build_summary(
        config=experiment_config,
        expected_seeds=seeds,
        results=list(results_by_seed.values()),
    )
    atomic_write_json(
        output_path,
        final_summary,
    )

    print("\nHGA evaluation finished.")
    print(
        "Average makespan:",
        final_summary["avg_makespan"],
    )
    print(
        "Worst makespan:",
        final_summary["worst_makespan"],
    )
    print(
        "Average solve time:",
        final_summary[
            "avg_solve_time_seconds"
        ],
    )
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()