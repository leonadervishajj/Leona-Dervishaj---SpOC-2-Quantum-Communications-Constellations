

import argparse
import csv
from pathlib import Path

from constellation import udp, combine_scores
from src.nsga2 import sort_fronts
from submission_helper import create_submission


RESULTS = Path("results")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export validated NSGA-II solutions for ESA"
    )

    parser.add_argument(
        "--run-dir",
        default=str(RESULTS),
        help="Folder containing nsga2_final_population.csv"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing JSON file"
    )

    args = parser.parse_args(argv)

    input_path = Path(args.run_dir) / "nsga2_final_population.csv"
    output_path = (
        Path(args.output)
        if args.output
        else Path(args.run_dir) / "esa_submission.json"
    )

    if not input_path.is_file():
        parser.error(f"CSV file not found: {input_path}")

    if output_path.exists() and not args.overwrite:
        parser.error(
            f"JSON already exists: {output_path}. "
            "Use another --output or --overwrite."
        )

    if not output_path.parent.is_dir():
        parser.error(
            f"Output folder does not exist: {output_path.parent}"
        )

    solutions = []

    with input_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            j1 = float(row["J1"])
            j2 = float(row["J2"])
            c1 = float(row["C1"])
            c2 = float(row["C2"])

            if c1 > 0 or c2 > 0:
                continue

            x = []
            for i in range(20):
                value = float(row[f"x{i}"])

                if i >= 10 and not value.is_integer():
                    raise ValueError(
                        f"x{i} must be an integer, got {value}"
                    )

                x.append(value if i < 10 else int(value))

            solutions.append({
                "x": x,
                "J1": j1,
                "J2": j2,
                "C1": c1,
                "C2": c2
            })

    if not solutions:
        raise ValueError("Nuk ka zgjidhje te vlefshme.")

    fronts = sort_fronts(solutions)
    pareto = [solutions[i] for i in fronts[0]]

    unique = []
    seen = set()

    for solution in pareto:
        key = tuple(solution["x"])

        if key not in seen:
            seen.add(key)
            unique.append(solution)

    unique = unique[:100]

    lb, ub = udp.get_bounds()

    print("\n--- VALIDIMI I KONFIGURIMEVE ESA ---", flush=True)

    for i, solution in enumerate(unique, start=1):
        x = solution["x"]

        if len(x) != 20:
            raise ValueError(f"Konfigurimi {i}: priten 20 parametra.")

        if not all(lb[k] <= x[k] <= ub[k] for k in range(20)):
            raise ValueError(f"Konfigurimi {i}: parametra jashte kufijve.")

        if not all(isinstance(x[k], int) for k in range(10, 20)):
            raise ValueError(f"Konfigurimi {i}: parametrat diskrete.")

        j1, j2, c1, c2 = udp.fitness(x)

        if c1 > 0 or c2 > 0:
            raise ValueError(
                f"Konfigurimi {i} nuk eshte i vlefshem: "
                f"C1={c1:.2f}, C2={c2:.2f}"
            )

        solution["J1"] = j1
        solution["J2"] = j2
        solution["C1"] = c1
        solution["C2"] = c2

        print(
            f"Konfigurimi {i}/{len(unique)}: "
            f"J1={j1:.4f}, J2={j2:.4f}, "
            f"C1={c1:.2f}, C2={c2:.2f} - VALID",
            flush=True
        )

    # Rillogarit Pareto pasi vlerat jane validuar nga ESA
    fronts = sort_fronts(unique)
    unique = [unique[i] for i in fronts[0]]

    print("\nTe gjitha konfigurimet e eksportuara jane te vlefshme!")

    x_vectors = [solution["x"] for solution in unique]

    create_submission(
        "spoc-2-quantum-communications-constellations",
        "quantum-communications-constellations",
        x_vectors,
        str(output_path),
        "NSGA-II Quantum Communications",
        "Pareto solutions generated and validated using NSGA-II"
    )

    points = [
        [s["J1"], s["J2"], s["C1"], s["C2"]]
        for s in unique
    ]

    print("Konfigurime Pareto:", len(unique))
    print("ESA Score:", combine_scores(points))
    print("JSON saved:", output_path.resolve())


if __name__ == "__main__":
    main()
