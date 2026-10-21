
import random

from constellation import udp
from src.selection import (
    crowding_distance,
    binary_tournament_selection
)
from src.genetic_operators import crossover, mutation


def generate_chromosome(lb, ub, rng):
    """Gjeneron 20 parametra sipas kufijve te ESA-s."""
    x = []

    for i, (low, high) in enumerate(zip(lb, ub)):
        if i < 10:
            x.append(rng.uniform(low, high))
        else:
            x.append(rng.randint(int(low), int(high)))

    return x


def evaluate(x):
    """Vlereson konfigurimin me modelin ESA."""
    j1, j2, c1, c2 = udp.fitness(x)

    return {
        "x": list(x),
        "J1": float(j1),
        "J2": float(j2),
        "C1": float(c1),
        "C2": float(c2),
        "valid": c1 <= 0 and c2 <= 0
    }


def violation(z):
    """Shuma e shkeljeve pozitive te kufizimeve."""
    return max(0.0, z["C1"]) + max(0.0, z["C2"])


def dominates(a, b):
    """
    Constraint-domination:
    - Zgjidhjet e vlefshme kane perparesi.
    - Mes dy te pavlefshmeve preferohet
      shkelja me e vogel.
    - Mes dy te vlefshmeve perdoret Pareto.
    """
    va = violation(a)
    vb = violation(b)

    if va == 0 and vb > 0:
        return True

    if vb == 0 and va > 0:
        return False

    if va > 0 and vb > 0:
        return va < vb

    return (
        a["J1"] <= b["J1"]
        and a["J2"] <= b["J2"]
        and (
            a["J1"] < b["J1"]
            or a["J2"] < b["J2"]
        )
    )


def sort_fronts(population):
    """Ndan popullaten ne fronte sipas dominimit."""
    remaining = list(range(len(population)))
    fronts = []

    while remaining:
        front = []

        for i in remaining:
            if not any(
                dominates(population[j], population[i])
                for j in remaining
                if j != i
            ):
                front.append(i)

        if not front:
            raise RuntimeError("Nuk u gjet front Pareto.")

        fronts.append(front)
        front_set = set(front)

        remaining = [
            i for i in remaining if i not in front_set
        ]

    return fronts


def rank_and_distance(population):
    """Kthen rangjet dhe crowding distance per secilin."""
    ranks = [0] * len(population)
    distances = [0.0] * len(population)

    fronts = sort_fronts(population)

    for rank, indices in enumerate(fronts):
        front = [population[i] for i in indices]
        crowding = crowding_distance(front)

        for i, distance in zip(indices, crowding):
            ranks[i] = rank
            distances[i] = distance

    return fronts, ranks, distances


def select_next_population(combined, size):
    """Elitist survival selection e NSGA-II."""
    fronts, _, _ = rank_and_distance(combined)
    survivors = []

    for indices in fronts:
        front = [combined[i] for i in indices]
        free_places = size - len(survivors)

        if free_places <= 0:
            break

        if len(front) <= free_places:
            survivors.extend(front)
        else:
            distances = crowding_distance(front)
            order = sorted(
                range(len(front)),
                key=lambda i: distances[i],
                reverse=True
            )
            survivors.extend(
                front[i] for i in order[:free_places]
            )
            break

    return survivors


def run_nsga2(population_size=20, generations=5, seed=42):
    """Ekzekuton NSGA-II per konstelacionet e ESA-s."""
    if population_size < 2 or population_size % 2 != 0:
        raise ValueError(
            "Population size duhet te jete cift dhe >= 2."
        )

    if generations < 1:
        raise ValueError("Generations duhet te jete >= 1.")

    rng = random.Random(seed)
    lb, ub = udp.get_bounds()

    population = [
        evaluate(generate_chromosome(lb, ub, rng))
        for _ in range(population_size)
    ]

    history = []

    for generation in range(1, generations + 1):
        _, ranks, distances = rank_and_distance(population)

        offspring = []

        while len(offspring) < population_size:
            parent1 = binary_tournament_selection(
                population, ranks, distances, rng
            )
            parent2 = binary_tournament_selection(
                population, ranks, distances, rng
            )

            child1, child2 = crossover(
                parent1["x"], parent2["x"], lb, ub, rng
            )

            child1 = mutation(
                child1, lb, ub, probability=0.1, rng=rng
            )
            child2 = mutation(
                child2, lb, ub, probability=0.1, rng=rng
            )

            offspring.append(evaluate(child1))
            offspring.append(evaluate(child2))

        combined = population + offspring

        population = select_next_population(
            combined, population_size
        )

        feasible = [z for z in population if z["valid"]]

        record = {
            "generation": generation,
            "feasible": len(feasible),
            "best_J1": (
                min(z["J1"] for z in feasible)
                if feasible else None
            ),
            "best_J2": (
                min(z["J2"] for z in feasible)
                if feasible else None
            )
        }
        history.append(record)

        print(
            f"Generation {generation}/{generations} | "
            f"Feasible: {len(feasible)}/{population_size} | "
            f"Best J1: {record['best_J1']} | "
            f"Best J2: {record['best_J2']}",
            flush=True
        )

    return population, history


if __name__ == "__main__":
    final_population, history = run_nsga2(
        population_size=20,
        generations=5,
        seed=42
    )

    feasible = [
        z for z in final_population if z["valid"]
    ]

    print("\n--- REZULTATET FINALE ---")
    print("Population:", len(final_population))
    print("Feasible:", len(feasible))

    if feasible:
        fronts = sort_fronts(feasible)
        pareto = [feasible[i] for i in fronts[0]]

        print("Pareto solutions:", len(pareto))

        for z in pareto:
            print(
                f"J1={z['J1']:.4f}, "
                f"J2={z['J2']:.4f}, "
                f"C1={z['C1']:.2f}, "
                f"C2={z['C2']:.2f}"
            )
