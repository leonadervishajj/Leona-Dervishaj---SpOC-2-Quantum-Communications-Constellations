
import json
import csv
import random
import math
from pathlib import Path

from constellation import udp
from pygmo import hypervolume

SOURCE = Path("submission/esa_submission_100.json")
OUT = Path("results/pareto_local_search")

TRIALS = 300
SEED = 2026
REF = [1.2, 1.4]

rng = random.Random(SEED)
lower, upper = udp.get_bounds()

def evaluate(x):
    f = udp.fitness(x)

    if not all(math.isfinite(float(v)) for v in f):
        return None

    if any(float(c) > 0 for c in f[2:]):
        return None

    return {
        "x": list(x),
        "f": [float(f[0]), float(f[1])]
    }

def get_hv(solutions):
    points = [
        s["f"] for s in solutions
        if s["f"][0] < REF[0] and s["f"][1] < REF[1]
    ]

    if not points:
        return 0.0

    return float(hypervolume(points).compute(REF))

def select_best(solutions, limit=100):
    # Largo konfigurimet identike
    unique = {}
    for s in solutions:
        unique[tuple(s["x"])] = s

    selected = list(unique.values())

    # Ruajme deri ne 100 konfigurime.
    # Heqim ato me kontributin me te vogel ne HV.
    while len(selected) > limit:
        current_hv = get_hv(selected)
        losses = []

        for i in range(len(selected)):
            without = selected[:i] + selected[i+1:]
            losses.append(current_hv - get_hv(without))

        idx = min(range(len(losses)), key=lambda i: losses[i])
        selected.pop(idx)

    return selected

def save_csv(path, solutions):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["J1", "J2"] + [f"x{i}" for i in range(20)]
        )
        for s in solutions:
            writer.writerow(s["f"] + s["x"])

if not SOURCE.exists():
    raise FileNotFoundError(f"Mungon: {SOURCE}")

data = json.loads(SOURCE.read_text(encoding="utf-8"))
payload = data[0] if isinstance(data, list) else data
vectors = payload["decisionVector"]

archive = []
for x in vectors:
    result = evaluate(x)
    if result is not None:
        archive.append(result)

if not archive:
    raise RuntimeError("Nuk u gjet asnje konfigurim valid.")

best = select_best(archive)
baseline_hv = get_hv(best)
best_hv = baseline_hv

OUT.mkdir(parents=True, exist_ok=True)
save_csv(OUT / "baseline.csv", best)

print("Konfigurime:", len(best), flush=True)
print(f"HV fillestar: {baseline_hv:.9f}", flush=True)

for trial in range(1, TRIALS + 1):
    parent = rng.choice(best)
    child = parent["x"].copy()

    # Mutacion i vogel ne 1 ose 2 parametra
    indices = rng.sample(range(20), rng.randint(1, 2))

    for i in indices:
        lo = float(lower[i])
        hi = float(upper[i])

        if i in (10, 11, 12, 13, 14, 15, 16, 17, 18, 19):
            child[i] = int(
                min(hi, max(lo, round(child[i] + rng.choice([-1, 1]))))
            )
        else:
            sigma = (hi - lo) * rng.choice([0.001, 0.005, 0.01])
            child[i] = min(
                hi,
                max(lo, child[i] + rng.gauss(0, sigma))
            )

    candidate = evaluate(child)

    if candidate is not None:
        proposed = select_best(best + [candidate])
        proposed_hv = get_hv(proposed)

        if proposed_hv > best_hv + 1e-10:
            best = proposed
            best_hv = proposed_hv

            save_csv(OUT / "best_population.csv", best)

            print(
                f"PERMIRESIM Trial {trial}: HV={best_hv:.9f}",
                flush=True
            )

    if trial % 25 == 0:
        save_csv(OUT / "checkpoint.csv", best)
        print(
            f"Trial {trial}/{TRIALS} | HV={best_hv:.9f}",
            flush=True
        )

if best_hv > baseline_hv + 1e-10:
    new_payload = dict(payload)
    new_payload["decisionVector"] = [s["x"] for s in best]

    output = [new_payload] if isinstance(data, list) else new_payload

    destination = OUT / "esa_submission_improved.json"
    destination.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8"
    )

    print("Submission i ri:", destination)

print("\nREZULTATI FINAL")
print(f"HV fillestar: {baseline_hv:.9f}")
print(f"HV final:     {best_hv:.9f}")
print(f"Permiresimi:  {best_hv - baseline_hv:+.9f}")
