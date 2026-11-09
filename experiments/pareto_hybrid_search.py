
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import pygmo as pg

from constellation import udp

# PARAMETRAT
SEED = 2026
TRIALS = 600
LIMIT = 100
REF = [1.2, 1.4]

SOURCE = Path("submission/esa_submission_100.json")
PREVIOUS = Path("results/pareto_local_search/best_population.csv")
OUT = Path("results/pareto_hybrid_search")

rng = random.Random(SEED)
np_rng = np.random.default_rng(SEED)

lower, upper = udp.get_bounds()
lower = np.asarray(lower, dtype=float)
upper = np.asarray(upper, dtype=float)

# Sipas struktures se kromozomit ESA:
# 10 parametra te vazhdueshem + 10 diskrete
INTEGER_INDICES = range(10, 20)


def evaluate(x):
    values = np.asarray(x, dtype=float).tolist()
    fitness = udp.fitness(values)

    if not all(math.isfinite(float(v)) for v in fitness):
        return None

    if any(float(c) > 0 for c in fitness[2:]):
        return None

    return {
        "x": values,
        "f": [float(fitness[0]), float(fitness[1])]
    }


def hypervolume(population):
    points = [
        item["f"]
        for item in population
        if item["f"][0] < REF[0]
        and item["f"][1] < REF[1]
    ]

    if not points:
        return 0.0

    return float(pg.hypervolume(points).compute(REF))


def select_best(population):
    unique = {
        tuple(item["x"]): item
        for item in population
    }

    population = list(unique.values())

    while len(population) > LIMIT:
        total_hv = hypervolume(population)

        contributions = [
            total_hv - hypervolume(
                population[:i] + population[i + 1:]
            )
            for i in range(len(population))
        ]

        remove_index = int(np.argmin(contributions))
        population.pop(remove_index)

    return population


def save_result(population, score):
    result = dict(payload)
    result["decisionVector"] = [
        item["x"] for item in population
    ]

    output = (
        [result]
        if isinstance(original, list)
        else result
    )

    destination = OUT / "esa_submission_best.json"
    temporary = OUT / "esa_submission_best.tmp"

    temporary.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8"
    )
    temporary.replace(destination)

    (OUT / "best_hv.txt").write_text(
        f"{score:.12f}",
        encoding="utf-8"
    )


# LEXIMI I SUBMISSION ORIGJINAL
if not SOURCE.exists():
    raise FileNotFoundError(f"Mungon: {SOURCE}")

original = json.loads(
    SOURCE.read_text(encoding="utf-8")
)

payload = (
    original[0]
    if isinstance(original, list)
    else original
)

archive = []

# NISEMI NGA LOCAL SEARCH
if PREVIOUS.exists():
    with PREVIOUS.open(
        newline="", encoding="utf-8"
    ) as file:

        for row in csv.DictReader(file):
            x = [
                float(row[f"x{i}"])
                for i in range(20)
            ]

            archive.append({
                "x": x,
                "f": [
                    float(row["J1"]),
                    float(row["J2"])
                ]
            })

    print("Burimi: Pareto Local Search")

else:
    print("Burimi: Submission origjinal")

    for x in payload["decisionVector"]:
        candidate = evaluate(x)

        if candidate is not None:
            archive.append(candidate)


if not archive:
    raise RuntimeError(
        "Nuk ka konfigurime te vlefshme."
    )

archive = select_best(archive)

# Verifikojme burimin me modelin ESA
for item in archive:
    check = evaluate(item["x"])

    if check is None or not np.allclose(
        check["f"],
        item["f"],
        rtol=0,
        atol=1e-7
    ):
        raise RuntimeError(
            "Fitness nuk perputhet me modelin ESA!"
        )

initial_hv = hypervolume(archive)
best_hv = initial_hv

OUT.mkdir(parents=True, exist_ok=True)
save_result(archive, best_hv)

print("Konfigurime:", len(archive))
print(f"HV FILLESTAR: {best_hv:.9f}", flush=True)

accepted = 0
valid_count = 0

try:
    for trial in range(1, TRIALS + 1):

        parent = rng.choice(archive)
        child = np.asarray(
            parent["x"], dtype=float
        ).copy()

        mode = trial % 3

        if mode == 0:
            # Mutacion me 1-4 parametra
            indices = rng.sample(
                range(20),
                rng.randint(1, 4)
            )

            scale = rng.choice([
                0.001, 0.005, 0.02, 0.05
            ])

            for i in indices:
                child[i] += rng.gauss(
                    0,
                    scale * (upper[i] - lower[i])
                )

        elif mode == 1:
            # Differential Evolution
            a, b = rng.sample(archive, 2)

            difference = (
                np.asarray(a["x"])
                - np.asarray(b["x"])
            )

            mask = np_rng.random(20) < 0.35
            mask[rng.randrange(20)] = True

            factor = rng.choice([
                0.25, 0.5, 0.8
            ])

            child[mask] += (
                factor * difference[mask]
            )

        else:
            # Ndryshime diskrete
            indices = rng.sample(
                list(INTEGER_INDICES),
                rng.randint(1, 3)
            )

            for i in indices:
                child[i] += rng.choice([
                    -3, -2, -1, 1, 2, 3
                ])

            i = rng.randrange(10)
            child[i] += rng.gauss(
                0,
                0.003 * (upper[i] - lower[i])
            )

        child = np.clip(
            child, lower, upper
        )

        # Parametrat diskrete behen integer
        child[10:] = np.rint(child[10:])

        candidate = evaluate(child)

        if candidate is not None:
            valid_count += 1

            proposed = select_best(
                archive + [candidate]
            )

            new_hv = hypervolume(proposed)

            if new_hv > best_hv + 1e-10:
                archive = proposed
                best_hv = new_hv
                accepted += 1

                save_result(
                    archive, best_hv
                )

                print(
                    f"PERMIRESIM Trial {trial}: "
                    f"HV={best_hv:.9f}",
                    flush=True
                )

        if trial % 25 == 0:
            print(
                f"Trial {trial}/{TRIALS} | "
                f"HV={best_hv:.9f} | "
                f"Valid={valid_count} | "
                f"Permiresime={accepted}",
                flush=True
            )

except KeyboardInterrupt:
    print(
        "Testimi u nderpre. "
        "Rezultati me i mire u ruajt."
    )

finally:
    save_result(archive, best_hv)

    print("\n--- REZULTATI FINAL ---")
    print(f"Fillestar: {initial_hv:.9f}")
    print(f"Final:     {best_hv:.9f}")
    print(
        f"Permiresimi: "
        f"{best_hv - initial_hv:+.9f}"
    )
    print(
        "Fajlli:",
        OUT / "esa_submission_best.json"
    )
