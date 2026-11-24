
import argparse
import json
import math

import numpy as np

from common import (
    lexo_submission,
    vlereso,
    udp,
    ROOT
)

# Parametrat e vektorit ESA
ORBITAL = list(range(0, 10))
ROVER = list(range(16, 20))

parser = argparse.ArgumentParser()

parser.add_argument("submission")

parser.add_argument(
    "--pika",
    type=int,
    nargs="+",
    required=True
)

parser.add_argument(
    "--evals",
    type=int,
    default=3000
)

parser.add_argument(
    "--seed",
    type=int,
    default=1
)

parser.add_argument(
    "--T0",
    type=float,
    default=0.001
)

args = parser.parse_args()

if args.evals < 1:
    parser.error("--evals duhet te jete >= 1")

rng = np.random.default_rng(args.seed)

lower, upper = [
    np.asarray(v, dtype=float)
    for v in udp.get_bounds()
]

vectors, _ = lexo_submission(args.submission)

if len(vectors) != 100:
    raise RuntimeError(
        "Submission duhet te kete 100 pika."
    )

if not all(
    0 <= i < len(vectors)
    for i in args.pika
):
    raise RuntimeError(
        "Indeksi i pikes eshte jashte kufijve."
    )

output_dir = ROOT / "results" / "hv_fiks_n"

output = (
    output_dir
    / f"kandidate_seed{args.seed}.json"
)

if output.exists():
    raise FileExistsError(
        f"Rezultati ekziston: {output}"
    )

candidates = []

for index in args.pika:
    original = np.asarray(
        vectors[index],
        dtype=float
    )

    J1_original, J2_original, feasible = (
        vlereso(original)
    )

    if not feasible:
        print(f"Pika {index} eshte e pavlefshme.")
        continue

    current = original.copy()
    current_J1 = J1_original

    best = original.copy()
    best_J1 = J1_original

    sigma = 0.05
    accepted = 0

    for step in range(args.evals):
        temperature = (
            args.T0
            * (1 - step / args.evals)
            + 1e-12
        )

        candidate = current.copy()

        # 25%: ndrysho pozicionin e nje roveri
        if rng.random() < 0.25:
            k = int(rng.choice(ROVER))

            candidate[k] = rng.integers(
                int(lower[k]),
                int(upper[k]) + 1
            )

        # 75%: ndrysho parametrat orbitale
        else:
            count = int(rng.integers(1, 4))

            indices = rng.choice(
                ORBITAL,
                size=count,
                replace=False
            )

            candidate[indices] += (
                rng.normal(
                    0,
                    sigma,
                    size=count
                )
                * (upper[indices] - lower[indices])
            )

        candidate = np.clip(
            candidate,
            lower,
            upper
        )

        J1, J2, feasible = vlereso(candidate)

        if not feasible:
            continue

        if J2 > J2_original + 1e-12:
            continue

        difference = J1 - current_J1

        if (
            difference < 0
            or rng.random()
            < math.exp(-difference / temperature)
        ):
            current = candidate
            current_J1 = J1

            accepted += 1

            if J1 < best_J1:
                best = candidate.copy()
                best_J1 = J1

                candidates.append(best.tolist())

        if (step + 1) % 200 == 0:
            if accepted > 60:
                sigma *= 1.3
            else:
                sigma *= 0.7

            sigma = float(
                np.clip(sigma, 1e-4, 0.3)
            )

            accepted = 0

    print(
        f"Pika {index}: "
        f"J2={J2_original:.4f} | "
        f"J1 {J1_original:.6f} -> {best_J1:.6f} | "
        f"fitimi={J1_original - best_J1:+.9e}"
    )

    candidates.append(best.tolist())

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

with output.open("x", encoding="utf-8") as file:
    json.dump(candidates, file, indent=2)

print(
    f"U ruajten {len(candidates)} kandidate "
    f"ne {output}"
)
