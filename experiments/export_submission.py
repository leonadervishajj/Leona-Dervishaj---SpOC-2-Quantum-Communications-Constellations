
import csv
import json
from pathlib import Path

from constellation import udp, combine_scores
from src.nsga2 import sort_fronts
from submission_helper import create_submission

RESULTS = Path("results")
INPUT = RESULTS / "nsga2_final_population.csv"
OUTPUT = RESULTS / "esa_submission.json"


def main():
    solutions = []

    with INPUT.open(newline="", encoding="utf-8") as file:
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

    # Heqim kromozomet identike
    unique = []
    seen = set()

    for solution in pareto:
        key = tuple(solution["x"])
        if key not in seen:
            seen.add(key)
            unique.append(solution)

    unique = unique[:100]

    lb, ub = udp.get_bounds()

    for solution in unique:
        x = solution["x"]

        assert len(x) == 20
        assert all(lb[i] <= x[i] <= ub[i] for i in range(20))
        assert all(isinstance(x[i], int) for i in range(10, 20))

            
    # Kontrollimi i konfigurimeve me modelin origjinal te ESA-s
    print("\n--- VALIDIMI I KONFIGURIMEVE ESA ---", flush=True)

    for i, solution in enumerate(unique, start=1):
        x = solution["x"]

        j1, j2, c1, c2 = udp.fitness(x)

        if c1 > 0 or c2 > 0:
            raise ValueError(
                f"Konfigurimi {i} nuk eshte i vlefshem: "
                f"C1={c1:.2f}, C2={c2:.2f}"
            )

        # Perditesojme objektivat me vlerat e rivleresuara
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

    print("\nTe gjitha konfigurimet jane te vlefshme!")

  
    # Krijimi i fajllit te dorezimit sipas ESA-s
    x = [solution["x"] for solution in unique]

    create_submission(
        "spoc-2-quantum-communications-constellations",
        "quantum-communications-constellations",
        x,
        str(OUTPUT),
        "NSGA-II Quantum Communications",
        "Pareto solutions generated and validated using NSGA-II"
    )


    points = [
        [s["J1"], s["J2"], s["C1"], s["C2"]]
        for s in unique
    ]

    print("Konfigurime Pareto:", len(unique))
    print("ESA Score:", combine_scores(points))
    print("JSON saved:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
