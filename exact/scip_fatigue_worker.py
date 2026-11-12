import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.instance_generator import instance_from_dict
from exact.scip_fatigue_solver import SCIPFatigueSolver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    instance = instance_from_dict(payload["instance"])
    solver = SCIPFatigueSolver(
        instance=instance,
        time_limit=payload["time_limit_seconds"],
        num_threads=payload["num_threads"],
        relative_gap=payload["relative_gap"],
        log_output=payload["log_output"],
    )
    result = solver.solve()
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
