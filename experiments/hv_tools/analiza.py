
import argparse
from collections import Counter

import numpy as np

from common import (
    lexo_submission,
    vlereso,
    hv,
    kontributet,
    jo_te_dominuara,
    REF,
    hv_projektit,
)

parser = argparse.ArgumentParser()

parser.add_argument("submission")

parser.add_argument(
    "--expected-hv",
    type=float,
    default=0.639415104087
)

args = parser.parse_args()

xs, _ = lexo_submission(args.submission)

F = []
validity = []

for x in xs:
    J1, J2, feasible = vlereso(x)

    F.append(
        (J1, J2)
        if feasible
        else (np.inf, np.inf)
    )

    validity.append(feasible)

F = np.asarray(F, dtype=float)

local_hv = hv(F)

print("=" * 60)
print("ANALIZA E FRONTIT PARETO")
print("=" * 60)

print(f"HV total: {local_hv:.12f}")
print(f"HV i pritur: {args.expected_hv:.12f}")
print(f"REF: {REF}")

if len(xs) != 100:
    raise SystemExit(
        "STOP: Submission nuk ka 100 zgjidhje."
    )

if not all(validity):
    raise SystemExit(
        "STOP: Submission permban zgjidhje te pavlefshme."
    )

if not np.isclose(
    local_hv,
    args.expected_hv,
    atol=1e-10,
    rtol=0
):
    raise SystemExit(
        "STOP: HV nuk perputhet me rekordin!"
    )

# Krahasimi me HV e projektit origjinal
rows = [
    {
        "x": list(map(float, x)),
        "f": [float(f[0]), float(f[1])]
    }
    for x, f in zip(xs, F)
]

try:
    project_hv = float(hv_projektit(rows))

except Exception as exc:
    print(
        "SHENIM: HV e projektit nuk e pranoi formatin:",
        repr(exc)
    )
    print(
        "HV lokale perputhet me rekordin, "
        "por krahasimi i dyte nuk u perfundua."
    )

else:
    print(f"HV sipas projektit: {project_hv:.12f}")

    if not np.isclose(
        project_hv,
        local_hv,
        atol=1e-10,
        rtol=0
    ):
        raise SystemExit(
            "STOP: Dy implementimet e HV nuk perputhen."
        )

nd = jo_te_dominuara(F)

nd_set = set(nd)

contributions = kontributet(F)

lost = len(xs) - len(nd)

outside_ref = sum(
    1
    for f in F
    if not (f[0] < REF[0] and f[1] < REF[1])
)

print()
print("STATISTIKAT")
print(f"Zgjidhje: {len(xs)}")
print(f"Te pavlefshme: {validity.count(False)}")
print(f"Jashte REF: {outside_ref}")
print(f"Jo te dominuara: {len(nd)}")
print(f"VENDE TE HUMBURA: {lost}")

# Kontrollimi i J2
counts = Counter(
    round(float(f[1]), 9)
    for f in F
)

duplicates = {
    value: count
    for value, count in counts.items()
    if count > 1
}

print()
print(f"Vlera te ndryshme J2: {len(counts)}")

if duplicates:
    print("Vlera J2 te perseritura:")
    print(duplicates)

# Shfaqja e gjithe frontit
print()
print("FRONTI PARETO")

print(
    f"{'idx':>4} "
    f"{'J1':>12} "
    f"{'J2':>12} "
    f"{'kontributi':>14} "
    f"statusi"
)

for i in sorted(
    range(len(F)),
    key=lambda i: (F[i, 1], F[i, 0])
):
    status = (
        "Jo e dominuar"
        if i in nd_set
        else "E dominuar / jashte REF"
    )

    print(
        f"{i:>4} "
        f"{F[i, 0]:>12.6f} "
        f"{F[i, 1]:>12.6f} "
        f"{contributions[i]:>14.3e} "
        f"{status}"
    )

# Boshllëqet ndërmjet pikave
gaps = []

for a, b in zip(nd, nd[1:]):
    area = (
        (F[b, 0] - F[a, 0])
        * (F[a, 1] - F[b, 1])
    )

    gaps.append((area, a, b))

gaps.sort(reverse=True)

print()
print("10 BOSHLLËQET MË TË MËDHA")

for area, a, b in gaps[:10]:
    print(
        f"Mes pikes {a} dhe {b}: "
        f"zona={area:.9e}"
    )

# Pikat me kontribut më të vogël
weak = sorted(
    nd,
    key=lambda i: contributions[i]
)[:15]

print()
print("15 PIKAT ME KONTRIBUTIN ME TE VOGEL")
print(weak)

print()
print("ANALIZA PERFUNDOI ME SUKSES")
