
import json
import sys
import copy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from constellation import udp

if isinstance(udp, type):
    udp = udp()

from experiments.map_elites_esa import REF as _REF
from experiments.map_elites_esa import hv as hv_projektit

REF = np.asarray(_REF, dtype=float)


def vlereso(x, fast=False):
    """Kthen J1, J2 dhe nese kandidati eshte i vlefshem."""

    x = np.asarray(x, dtype=float)

    if x.shape != (20,) or not np.isfinite(x).all():
        return np.inf, np.inf, False

    f = np.asarray(
        udp.fitness(x.tolist()),
        dtype=float
    )

    if f.shape != (4,) or not np.isfinite(f).all():
        return np.inf, np.inf, False

    J1 = float(f[0])
    J2 = float(f[1])

    feasible = bool(np.all(f[2:] <= 0))

    return J1, J2, feasible


def lexo_submission(path):
    with open(path, encoding="utf-8") as file:
        document = json.load(file)

    payload = (
        document[0]
        if isinstance(document, list)
        and document
        and isinstance(document[0], dict)
        else document
    )

    if not isinstance(payload, dict):
        raise ValueError("Formati i submission-it nuk eshte ESA")

    vectors = payload["decisionVector"]

    return [
        list(map(float, x)) for x in vectors
    ], document


def shkruaj_submission(path, xs, shablloni):
    path = Path(path)

    if path.exists():
        raise FileExistsError(
            f"Fajlli ekziston: {path}"
        )

    if len(xs) != 100:
        raise ValueError(
            "Submission duhet te kete 100 zgjidhje"
        )

    if len({tuple(x) for x in xs}) != 100:
        raise ValueError(
            "Submission ka vektore te perseritur"
        )

    document = copy.deepcopy(shablloni)

    payload = (
        document[0]
        if isinstance(document, list)
        else document
    )

    payload["decisionVector"] = [
        list(map(float, x)) for x in xs
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open("x", encoding="utf-8") as file:
        json.dump(document, file, indent=2)


def jo_te_dominuara(F):
    """Indekset jo te dominuara per minimizim."""

    F = np.asarray(F, dtype=float)

    indices = [
        i for i in range(len(F))
        if np.all(np.isfinite(F[i]))
        and F[i, 0] < REF[0]
        and F[i, 1] < REF[1]
    ]

    indices.sort(
        key=lambda i: (F[i, 0], F[i, 1])
    )

    selected = []
    best_j2 = np.inf

    for i in indices:
        if F[i, 1] < best_j2:
            selected.append(i)
            best_j2 = F[i, 1]

    return selected


def hv(F):
    """Hypervolume 2D per dy objektiva qe minimizohen."""

    F = np.asarray(F, dtype=float)

    nd = jo_te_dominuara(F)

    total = 0.0

    for k, i in enumerate(nd):
        next_x = (
            F[nd[k + 1], 0]
            if k + 1 < len(nd)
            else REF[0]
        )

        total += (
            (next_x - F[i, 0])
            * (REF[1] - F[i, 1])
        )

    return float(total)


def kontributet(F):
    """Kontributi ekskluziv i seciles pike ne HV."""

    F = np.asarray(F, dtype=float)

    contributions = np.zeros(len(F))

    nd = jo_te_dominuara(F)

    for k, i in enumerate(nd):
        next_x = (
            F[nd[k + 1], 0]
            if k + 1 < len(nd)
            else REF[0]
        )

        previous_y = (
            F[nd[k - 1], 1]
            if k > 0
            else REF[1]
        )

        contributions[i] = (
            (next_x - F[i, 0])
            * (previous_y - F[i, 1])
        )

    return contributions
