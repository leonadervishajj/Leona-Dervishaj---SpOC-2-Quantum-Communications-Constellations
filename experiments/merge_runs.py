
import argparse
import csv
import math
from pathlib import Path

from constellation import udp, combine_scores
from src.nsga2 import sort_fronts


REF_POINT = (1.2, 1.4)


def _hypervolume_2d(points):
    """Llogarit Hypervolume per minimizimin e J1 dhe J2."""
    ref_j1, ref_j2 = REF_POINT

    filtered = sorted(
        (float(j1), float(j2))
        for j1, j2 in points
        if math.isfinite(j1)
        and math.isfinite(j2)
        and j1 < ref_j1
        and j2 < ref_j2
    )

    volume = 0.0
    best_j2 = ref_j2

    for j1, j2 in filtered:
        if j2 < best_j2:
            volume += (ref_j1 - j1) * (best_j2 - j2)
            best_j2 = j2

    return max(0.0, volume)


def _hv_contributions(points):
    """Kontributi individual ne Hypervolume."""
    total = _hypervolume_2d(points)
    contributions = []

    for i in range(len(points)):
        remaining = points[:i] + points[i + 1:]
        contribution = total - _hypervolume_2d(remaining)
        contributions.append(max(0.0, contribution))

    return contributions


def select_solutions(pareto, max_solutions=100):
    """Zgjedh deri ne max_solutions duke ruajtur Hypervolume."""
    if max_solutions < 1:
        raise ValueError("max_solutions duhet te jete >= 1")

    if len(pareto) <= max_solutions:
        return list(pareto)

    # Renditja siguron rezultate te riprodhueshme.
    selected = sorted(
        pareto,
        key=lambda s: (
            s["J1"],
            s["J2"],
            tuple(s["x"])
        )
    )

    while len(selected) > max_solutions:
        points = [(s["J1"], s["J2"]) for s in selected]
        contributions = _hv_contributions(points)

        # Largo zgjidhjen me kontributin me te vogel.
        index_to_remove = min(
            range(len(selected)),
            key=lambda i: (contributions[i], i)
        )

        selected.pop(index_to_remove)

    return selected


def main(argv=None):
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

    parser.add_argument(
        "--max-solutions",
        type=int,
        default=100
    )

    args = parser.parse_args(argv)

    if args.max_solutions < 1:
        parser.error("--max-solutions duhet te jete >= 1")

    output_dir = Path(args.output_dir)
    output_csv = output_dir / "nsga2_final_population.csv"

    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
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

        with csv_path.open(
            newline="",
            encoding="utf-8"
        ) as file:

            for row in csv.DictReader(file):
                x = []

                for i in range(20):
                    value = float(row[f"x{i}"])

                    if not math.isfinite(value):
                        raise ValueError("Non-finite parameter")

                    if i >= 10:
                        if not value.is_integer():
                            raise ValueError(
                                f"x{i} is not an integer"
                            )
                        value = int(value)

                    x.append(value)

                key = tuple(x)

                if key in seen:
                    continue

                seen.add(key)

                if not all(
                    lb[i] <= x[i] <= ub[i]
                    for i in range(20)
                ):
                    continue

                j1, j2, c1, c2 = udp.fitness(x)

                if not all(
                    math.isfinite(v)
                    for v in (j1, j2, c1, c2)
                ):
                    continue

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

    # Perzgjedhja automatike kur kemi mbi 100 zgjidhje
    selected = select_solutions(
        pareto,
        args.max_solutions
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    fields = (
        ["J1", "J2", "C1", "C2", "valid"]
        + [f"x{i}" for i in range(20)]
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )
        writer.writeheader()

        for solution in selected:
            row = {
                name: solution[name]
                for name in ["J1", "J2", "C1", "C2", "valid"]
            }

            for i, value in enumerate(solution["x"]):
                row[f"x{i}"] = value

            writer.writerow(row)

    points = [
        [s["J1"], s["J2"], s["C1"], s["C2"]]
        for s in selected
    ]

    score = float(combine_scores(points))
    hypervolume = max(0.0, -score / 10000.0)

    print("\n--- MERGED RESULTS ---")
    print("Feasible unique solutions:", len(solutions))
    print("Pareto solutions:", len(pareto))
    print("Selected solutions:", len(selected))
    print(f"Hypervolume: {hypervolume:.6f}")
    print(f"ESA Score: {score:.6f}")
    print("CSV saved:", output_csv.resolve())


if __name__ == "__main__":
    main()
