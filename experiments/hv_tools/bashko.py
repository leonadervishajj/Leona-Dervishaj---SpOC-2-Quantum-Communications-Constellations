
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from common import (
    lexo_submission,
    shkruaj_submission,
    vlereso,
    hv,
    kontributet,
    jo_te_dominuara
)

parser = argparse.ArgumentParser()

parser.add_argument("submission")

parser.add_argument(
    "kandidate",
    nargs="+"
)

parser.add_argument(
    "--out",
    default="results/hv_fiks_n/submission_i_ri.json"
)

args = parser.parse_args()

original_vectors, document = lexo_submission(
    args.submission
)

if len(original_vectors) != 100:
    raise RuntimeError(
        "Submission origjinal duhet te kete 100 pika."
    )

output = Path(args.out)

if output.exists():
    raise FileExistsError(
        f"Fajlli ekziston: {output}"
    )

# Verifiko baseline
original_fitness = []

for x in original_vectors:
    J1, J2, feasible = vlereso(x)

    if not feasible:
        raise RuntimeError(
            "Baseline ESA ka zgjidhje te pavlefshme."
        )

    original_fitness.append((J1, J2))

old_hv = hv(
    np.asarray(original_fitness, dtype=float)
)

# Lexo kandidatet
all_vectors = list(original_vectors)

for pattern in args.kandidate:
    matches = sorted(glob.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"Nuk u gjet: {pattern}"
        )

    for path in matches:
        with open(path, encoding="utf-8") as file:
            all_vectors.extend(json.load(file))

# Mbaj vektoret e ndryshem dhe te vlefshem
seen = set()
X = []
F = []

for vector in all_vectors:
    try:
        key = tuple(map(float, vector))

        if key in seen:
            continue

        seen.add(key)

        J1, J2, feasible = vlereso(key)

        if not feasible:
            continue

        X.append(list(key))
        F.append((J1, J2))

    except (TypeError, ValueError):
        continue

F = np.asarray(F, dtype=float)

if len(X) < 100:
    raise RuntimeError(
        "Nuk kemi 100 zgjidhje te vlefshme dhe te ndryshme."
    )

# Zgjidh pikat jo te dominuara
selected = jo_te_dominuara(F)

# Nese jane mbi 100, largo kontributin me te vogel
while len(selected) > 100:
    contributions = kontributet(F[selected])

    remove_position = int(
        np.argmin(contributions)
    )

    selected.pop(remove_position)

# Ploteso vendet e lira me vektore te ndryshem
final_vectors = [
    X[i] for i in selected
]

chosen = {
    tuple(x) for x in final_vectors
}

for x in X:
    if len(final_vectors) == 100:
        break

    key = tuple(x)

    if key not in chosen:
        final_vectors.append(x)
        chosen.add(key)

if len(final_vectors) != 100:
    raise RuntimeError(
        "Rezultati final nuk ka 100 pika."
    )

if len(chosen) != 100:
    raise RuntimeError(
        "Rezultati final ka dublikata."
    )

# Verifikimi final me fitness origjinal ESA
verified_fitness = []

for i, x in enumerate(final_vectors):
    J1, J2, feasible = vlereso(x)

    if not feasible:
        raise RuntimeError(
            f"Pika finale {i} eshte e pavlefshme."
        )

    verified_fitness.append((J1, J2))

new_hv = hv(
    np.asarray(verified_fitness, dtype=float)
)

print()
print("FINAL ESA VERIFICATION 100/100")
print(f"HV i vjeter: {old_hv:.12f}")
print(f"HV i ri: {new_hv:.12f}")
print(f"Permiresimi: {new_hv - old_hv:+.12f}")

if new_hv > old_hv + 1e-12:
    shkruaj_submission(
        output,
        final_vectors,
        document
    )

    print(f"NEW VERIFIED RECORD: {output}")

else:
    print(
        "NO IMPROVEMENT - baseline nuk ndryshohet."
    )
