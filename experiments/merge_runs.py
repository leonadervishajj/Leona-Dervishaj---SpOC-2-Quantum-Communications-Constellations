
import argparse
import csv
import math
from pathlib import Path

from constellation import udp, combine_scores
from src.nsga2 import sort_fronts


def main():
    parser = argparse.ArgumentParser(
        description="Merge NSGA-II results from multiple runs"
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Folders containing final population CSV files"
    )
    parser.add_argument(
        "--output-dir",
        default="results/merged_runs"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_csv = output_dir / "nsga2_final_population.csv"

    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(
            f"Output folder is not empty: {output_dir}"
        )

    solutions = []
    seen = set()
    lb, ub = udp.get_bounds()

    for folder in args.run_dirs:
        csv_path = Path(folder) / "nsga2_final_population.csv"

        if not csv_path.is_file():
            parser.error(f"File not found: {csv_path}")

        with csv_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                x = []

                for i in range(20):
                    value = float(row[f"x{i}"])
                    if not math.isfinite(value):
                        raise ValueError("Non-finite parameter")

                    if i >= 10:
                        if not value.is_integer():
                            raise ValueError(f"x{i} is not an integer")
                        value = int(value)

                    x.append(value)

                key = tuple(x)
                if key in seen:
                    continue
                seen.add(key)

                if not all(lb[i] <= x[i] <= ub[i] for i in range(20)):
                    continue

                j1, j2, c1, c2 = udp.fitness(x)

                if c1 > 0 or c2 > 0:
                    continue

                solutions.append({
                    "x": x,
                    "J1": float(j1),
                    "J2": float(j2),
                    "C1": float(c1),
                    "C2": float(c2),
                    "valid": True
                })

    if not solutions:
        raise ValueError("No feasible solutions found.")

    fronts = sort_fronts(solutions)
    pareto = [solutions[i] for i in fronts[0]]

    # Kufiri maksimal i konfigurimeve te dorezimit
    if len(pareto) > 100:
        raise ValueError(
            f"Found {len(pareto)} Pareto solutions. "
            "Selection to 100 is needed before exporting."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    fields = (
        ["J1", "J2", "C1", "C2", "valid"]
        + [f"x{i}" for i in range(20)]
    )

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for solution in pareto:
            row = {
                name: solution[name]
                for name in ["J1", "J2", "C1", "C2", "valid"]
            }
            for i, value in enumerate(solution["x"]):
                row[f"x{i}"] = value

            writer.writerow(row)

    points = [
        [s["J1"], s["J2"], s["C1"], s["C2"]]
        for s in pareto
    ]
    score = float(combine_scores(points))
    hypervolume = max(0.0, -score / 10000.0)

    print("\n--- MERGED RESULTS ---")
    print("Feasible unique solutions:", len(solutions))
    print("Pareto solutions:", len(pareto))
    print(f"Hypervolume: {hypervolume:.6f}")
    print(f"ESA Score: {score:.6f}")
    print("CSV saved:", output_csv.resolve())


if __name__ == "__main__":
    main()
