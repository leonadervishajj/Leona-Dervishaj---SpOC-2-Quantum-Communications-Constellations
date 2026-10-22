
import csv
import argparse
import statistics
from pathlib import Path

from src.nsga2 import run_nsga2, sort_fronts


def main():
    parser = argparse.ArgumentParser(
        description="ESA SpOC 2 - Multi-seed analysis"
    )

    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 2026]
    )

    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    records = []

    for seed in args.seeds:
        print(f"\n--- SEED {seed} ---", flush=True)

        population, history = run_nsga2(
            population_size=args.population,
            generations=args.generations,
            seed=seed
        )

        feasible = [z for z in population if z["valid"]]

        record = {
            "seed": seed,
            "feasible": len(feasible),
            "best_J1": "",
            "best_J2": "",
            "pareto_count": 0
        }

        if feasible:
            fronts = sort_fronts(feasible)
            pareto = [feasible[i] for i in fronts[0]]

            record["best_J1"] = min(z["J1"] for z in feasible)
            record["best_J2"] = min(z["J2"] for z in feasible)
            record["pareto_count"] = len(pareto)

        records.append(record)
        print("Result:", record, flush=True)

    output = results_dir / "multi_seed_summary.csv"

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed", "feasible", "best_J1",
                "best_J2", "pareto_count"
            ]
        )
        writer.writeheader()
        writer.writerows(records)

    valid = [r for r in records if r["feasible"] > 0]

    print("\n--- SUMMARY ---")

    if valid:
        for key in ("best_J1", "best_J2"):
            values = [r[key] for r in valid]

            print(f"Mean {key}: {statistics.mean(values):.6f}")

            if len(values) > 1:
                print(
                    f"Std {key}: "
                    f"{statistics.stdev(values):.6f}"
                )

    print("Saved:", output.resolve())


if __name__ == "__main__":
    main()
