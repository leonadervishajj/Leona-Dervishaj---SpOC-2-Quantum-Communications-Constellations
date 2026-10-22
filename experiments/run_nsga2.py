
import csv
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.nsga2 import run_nsga2, sort_fronts


def save_csv(path, rows, fieldnames):
    """Ruajtja e rezultateve ne format CSV."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="ESA SpOC 2 - NSGA-II experiments"
    )

    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Folderi ku do te ruhen rezultatet
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Ekzekutimi i algoritmit NSGA-II
    population, history = run_nsga2(
        population_size=args.population,
        generations=args.generations,
        seed=args.seed
    )

    # 1. Ruaj historine e gjeneratave
    save_csv(
        results_dir / "nsga2_history.csv",
        history,
        ["generation", "feasible", "best_J1", "best_J2"]
    )

    # 2. Ruaj popullaten finale me 20 parametrat
    final_rows = []

    for z in population:
        row = {
            "J1": z["J1"],
            "J2": z["J2"],
            "C1": z["C1"],
            "C2": z["C2"],
            "valid": z["valid"]
        }

        for i, value in enumerate(z["x"]):
            row[f"x{i}"] = value

        final_rows.append(row)

    fields = (
        ["J1", "J2", "C1", "C2", "valid"]
        + [f"x{i}" for i in range(20)]
    )

    save_csv(
        results_dir / "nsga2_final_population.csv",
        final_rows,
        fields
    )

    # 3. Grafiku i progresit per J1 dhe J2
    generations = [h["generation"] for h in history]
    best_j1 = [h["best_J1"] for h in history]
    best_j2 = [h["best_J2"] for h in history]

    plt.figure(figsize=(9, 5))

    plt.plot(
        generations,
        best_j1,
        marker="o",
        label="Best J1"
    )

    plt.plot(
        generations,
        best_j2,
        marker="s",
        label="Best J2"
    )

    plt.xlabel("Generation")
    plt.ylabel("Normalized objective value")
    plt.title("NSGA-II Optimization Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        results_dir / "nsga2_progress.png",
        dpi=200
    )
    plt.close()

    # 4. Grafiku final Pareto
    feasible = [
        z for z in population if z["valid"]
    ]

    if feasible:
        fronts = sort_fronts(feasible)
        pareto = [feasible[i] for i in fronts[0]]

        plt.figure(figsize=(9, 6))

        plt.scatter(
            [z["J1"] for z in feasible],
            [z["J2"] for z in feasible],
            alpha=0.45,
            label="Feasible population"
        )

        plt.scatter(
            [z["J1"] for z in pareto],
            [z["J2"] for z in pareto],
            label="Pareto front"
        )

        plt.xlabel("J1 - Communication cost")
        plt.ylabel("J2 - Infrastructure cost")
        plt.title("ESA SpOC 2 - Final Pareto Solutions")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            results_dir / "nsga2_pareto.png",
            dpi=200
        )
        plt.close()

    print("\n--- EXPERIMENT COMPLETED ---")
    print("Results saved in:", results_dir.resolve())


if __name__ == "__main__":
    main()
