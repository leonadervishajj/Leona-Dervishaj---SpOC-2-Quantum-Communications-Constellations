
import csv
from pathlib import Path

from constellation import udp
from src.nsga2 import sort_fronts


RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "nsga2_final_population.csv"


def main():
    # Konfigurimi fillestar zyrtar i ESA-s
    baseline_x = udp.example()
    baseline = udp.fitness(baseline_x)

    b_j1, b_j2, b_c1, b_c2 = baseline

    print("\n--- ESA BASELINE ---")
    print(f"J1: {b_j1:.6f}")
    print(f"J2: {b_j2:.6f}")
    print(f"C1: {b_c1:.2f}")
    print(f"C2: {b_c2:.2f}")

    if b_c1 > 0 or b_c2 > 0:
        print("Baseline nuk eshte i vlefshem.")
        return

    # Lexojme rezultatet finale te NSGA-II
    feasible = []

    with open(CSV_PATH, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            j1 = float(row["J1"])
            j2 = float(row["J2"])
            c1 = float(row["C1"])
            c2 = float(row["C2"])

            if c1 <= 0 and c2 <= 0:
                feasible.append({
                    "J1": j1,
                    "J2": j2,
                    "C1": c1,
                    "C2": c2
                })

    print("\n--- NSGA-II RESULTS ---")
    print("Feasible configurations:", len(feasible))

    if not feasible:
        print("Nuk ka konfigurime te vlefshme.")
        return

    fronts = sort_fronts(feasible)
    pareto = [feasible[i] for i in fronts[0]]

    print("Pareto configurations:", len(pareto))

    best_j1 = min(feasible, key=lambda z: z["J1"])
    best_j2 = min(feasible, key=lambda z: z["J2"])

    print("\n--- OBJECTIVE COMPARISON ---")

    print(f"Baseline J1: {b_j1:.6f}")
    print(f"Best NSGA-II J1: {best_j1['J1']:.6f}")

    print(f"Baseline J2: {b_j2:.6f}")
    print(f"Best NSGA-II J2: {best_j2['J2']:.6f}")

    # Kontrollojme dominimin ne te dy objektivat
    dominating = [
        z for z in pareto
        if z["J1"] <= b_j1
        and z["J2"] <= b_j2
        and (z["J1"] < b_j1 or z["J2"] < b_j2)
    ]

    print("\n--- BASELINE DOMINANCE ---")

    if dominating:
        print(
            "NSGA-II ka gjetur",
            len(dominating),
            "konfigurime Pareto qe dominojne baseline."
        )
    else:
        print(
            "Asnje konfigurim Pareto nuk dominon baseline."
        )


if __name__ == "__main__":
    main()
