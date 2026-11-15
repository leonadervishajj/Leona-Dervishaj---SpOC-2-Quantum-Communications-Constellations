
"""
ESA SpOC 2 - Quantum Communications Constellations

V4 - ADAPTIVE PARETO EVOLUTION

Strategjite:
1. Local Search
2. Pareto Gap Search
3. Differential Evolution
4. Global Exploration

Niset nga rezultati i verifikuar i V3.
Ruan automatikisht submission, checkpoint dhe CSV.
Nuk ndryshon algoritmet ose submission-et e vjetra.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pygmo as pg

from constellation import udp


# ==========================================================
# 1. KONFIGURIMI
# ==========================================================

REF = [1.2, 1.4]

MAX_POP = 100
DIM = 20

EXPECTED_MIN_HV = 0.611462157

SOURCE = Path(
    "submission/esa_submission_hv_v3.json"
)

# Nese ende nuk e kemi kopjuar V3 ne submission,
# perdorim rezultatin direkt nga results.
FALLBACK_SOURCE = Path(
    "results/pareto_hv_v3/esa_submission_best.json"
)

OUT = Path(
    "results/pareto_adaptive_v4"
)

BEST = OUT / "esa_submission_best.json"
STATE = OUT / "state.json"
LOG = OUT / "trials.csv"

OPERATOR_NAMES = [
    "local",
    "pareto_gap",
    "differential",
    "exploration"
]


# ==========================================================
# 2. LEXIMI DHE RUAJTJA E JSON
# ==========================================================

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            indent=2
        )

    temporary.replace(path)


def get_payload(document):
    if isinstance(document, list):
        if len(document) != 1:
            raise ValueError(
                "Submission duhet te kete nje objekt."
            )

        return document[0]

    return document


# ==========================================================
# 3. VLERESIMI SIPAS MODELiT ESA
# ==========================================================

def evaluate(vector):

    x = np.asarray(
        vector,
        dtype=float
    )

    if x.shape != (DIM,):
        return None

    if not np.all(np.isfinite(x)):
        return None

    try:
        result = np.asarray(
            udp.fitness(x.tolist()),
            dtype=float
        )

    except (ValueError, OverflowError, ArithmeticError):
        return None

    if result.size < 4:
        return None

    if not np.all(np.isfinite(result)):
        return None

    # C1, C2, ... duhet te jene <= 0
    if np.any(result[2:] > 0):
        return None

    return {
        "x": x.tolist(),
        "f": [
            float(result[0]),
            float(result[1])
        ]
    }


# ==========================================================
# 4. HYPERVOLUME
# ==========================================================

def hypervolume(population):

    points = []

    for item in population:

        f1, f2 = item["f"]

        if (
            f1 < REF[0]
            and f2 < REF[1]
        ):
            points.append([f1, f2])

    if not points:
        return 0.0

    return float(
        pg.hypervolume(points).compute(REF)
    )


# ==========================================================
# 5. HEQJA E DUPLIKATEVE
# ==========================================================

def remove_duplicates(population):

    seen = set()
    unique = []

    for item in population:

        key = tuple(item["x"])

        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


# ==========================================================
# 6. MBAJTJA E 100 KONFIGURIMEVE ME TE MIRA
# ==========================================================

def reduce_population(population):

    population = remove_duplicates(population)

    while len(population) > MAX_POP:

        current_hv = hypervolume(population)

        contributions = []

        for i in range(len(population)):

            remaining = (
                population[:i]
                + population[i + 1:]
            )

            hv_without = hypervolume(
                remaining
            )

            contribution = (
                current_hv - hv_without
            )

            contributions.append(
                contribution
            )

        remove_index = int(
            np.argmin(contributions)
        )

        population.pop(remove_index)

    return population


# ==========================================================
# 7. GJETJA E PARETO FRONT
# ==========================================================

def pareto_front(population):

    front = []

    for i, item in enumerate(population):

        f = np.asarray(
            item["f"],
            dtype=float
        )

        if np.any(f >= REF):
            continue

        dominated = False

        for j, other in enumerate(population):

            if i == j:
                continue

            g = np.asarray(
                other["f"],
                dtype=float
            )

            if (
                np.all(g <= f)
                and np.any(g < f)
            ):
                dominated = True
                break

        if not dominated:
            front.append(item)

    return front


# ==========================================================
# 8. ZGJEDHJA E PRINDERVE
# ==========================================================

def choose_parent(population, rng):

    front = pareto_front(population)

    if (
        front
        and rng.random() < 0.85
    ):
        index = int(
            rng.integers(len(front))
        )

        return front[index]

    index = int(
        rng.integers(len(population))
    )

    return population[index]


# ==========================================================
# 9. ZGJEDHJA E HAPESIRAVE NE PARETO FRONT
# ==========================================================

def choose_gap(population, rng):

    front = pareto_front(population)

    if len(front) < 2:
        return (
            choose_parent(population, rng),
            choose_parent(population, rng)
        )

    front.sort(
        key=lambda item: item["f"][0]
    )

    points = np.asarray(
        [item["f"] for item in front],
        dtype=float
    )

    differences = np.abs(
        np.diff(points, axis=0)
    )

    ranges = np.maximum(
        np.ptp(points, axis=0),
        1e-12
    )

    gaps = (
        differences[:, 0] / ranges[0]
        + differences[:, 1] / ranges[1]
        + 0.01
    )

    probabilities = gaps / gaps.sum()

    index = int(
        rng.choice(
            len(gaps),
            p=probabilities
        )
    )

    return (
        front[index],
        front[index + 1]
    )


# ==========================================================
# 10. KRIJIMI I KANDIDATEVE TE RINJ
# ==========================================================

def generate_candidate(
    population,
    mode,
    rng,
    lower,
    upper
):

    span = upper - lower

    # ------------------------------------------------------
    # MODE 0: LOCAL SEARCH
    # ------------------------------------------------------

    if mode == 0:

        parent = choose_parent(
            population,
            rng
        )

        child = np.asarray(
            parent["x"],
            dtype=float
        ).copy()

        indices = rng.choice(
            DIM,
            size=int(rng.integers(1, 5)),
            replace=False
        )

        scale = float(
            rng.choice([
                0.0005,
                0.002,
                0.008,
                0.02
            ])
        )

        child[indices] += (
            rng.normal(
                0,
                scale,
                len(indices)
            ) * span[indices]
        )

    # ------------------------------------------------------
    # MODE 1: PARETO GAP SEARCH
    # ------------------------------------------------------

    elif mode == 1:

        parent_a, parent_b = choose_gap(
            population,
            rng
        )

        a = np.asarray(
            parent_a["x"],
            dtype=float
        )

        b = np.asarray(
            parent_b["x"],
            dtype=float
        )

        alpha = float(
            rng.uniform(0.1, 0.9)
        )

        child = (
            alpha * a
            + (1.0 - alpha) * b
        )

        # Ruajme kombinime diskrete nga prinderit
        mask = rng.random(10) < 0.5

        child[10:] = np.where(
            mask,
            a[10:],
            b[10:]
        )

        indices = rng.choice(
            10,
            size=int(rng.integers(1, 5)),
            replace=False
        )

        child[indices] += (
            rng.normal(
                0,
                0.02,
                len(indices)
            ) * span[indices]
        )

    # ------------------------------------------------------
    # MODE 2: DIFFERENTIAL EVOLUTION
    # ------------------------------------------------------

    elif mode == 2:

        indices = rng.choice(
            len(population),
            size=3,
            replace=False
        )

        a, b, c = [
            np.asarray(
                population[int(i)]["x"],
                dtype=float
            )
            for i in indices
        ]

        factor = float(
            rng.uniform(0.25, 1.0)
        )

        donor = (
            a + factor * (b - c)
        )

        child = a.copy()

        mask = rng.random(DIM) < 0.55

        mask[
            int(rng.integers(DIM))
        ] = True

        child[mask] = donor[mask]

    # ------------------------------------------------------
    # MODE 3: GLOBAL EXPLORATION
    # ------------------------------------------------------

    else:

        parent = choose_parent(
            population,
            rng
        )

        child = np.asarray(
            parent["x"],
            dtype=float
        ).copy()

        continuous_indices = rng.choice(
            10,
            size=int(rng.integers(2, 6)),
            replace=False
        )

        for i in continuous_indices:

            if rng.random() < 0.20:

                child[i] = rng.uniform(
                    lower[i],
                    upper[i]
                )

            else:

                scale = rng.uniform(
                    0.03,
                    0.20
                )

                child[i] += rng.normal(
                    0,
                    scale * span[i]
                )

        discrete_indices = rng.choice(
            np.arange(10, 20),
            size=int(rng.integers(1, 5)),
            replace=False
        )

        for i in discrete_indices:

            if rng.random() < 0.30:

                child[i] = rng.integers(
                    math.ceil(lower[i]),
                    math.floor(upper[i]) + 1
                )

            else:

                child[i] += int(
                    rng.choice([
                        -5, -3, -2, -1,
                        1, 2, 3, 5
                    ])
                )

    # ------------------------------------------------------
    # RESPEKTIMI I KUFIJVE
    # ------------------------------------------------------

    child = np.clip(
        child,
        lower,
        upper
    )

    # Parametrat diskrete jane ne pozitat 10-19
    child[10:] = np.rint(
        child[10:]
    )

    return child


# ==========================================================
# 11. KRIJIMI I SUBMISSION
# ==========================================================

def create_submission(
    original_document,
    population
):

    payload = dict(
        get_payload(original_document)
    )

    payload["decisionVector"] = [
        item["x"]
        for item in population
    ]

    if isinstance(original_document, list):
        return [payload]

    return payload


# ==========================================================
# 12. RUAJTJA E CHECKPOINT
# ==========================================================

def save_checkpoint(
    original_document,
    population,
    trial,
    attempts,
    successes,
    rng
):

    score = hypervolume(
        population
    )

    write_json(
        BEST,
        create_submission(
            original_document,
            population
        )
    )

    write_json(
        STATE,
        {
            "trial": trial,
            "hypervolume": score,
            "population": population,
            "attempts": attempts,
            "successes": successes,
            "rng_state": rng.bit_generator.state
        }
    )


# ==========================================================
# 13. PROGRAMI KRYESOR
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trials",
        type=int,
        default=1200
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    args = parser.parse_args()

    if args.trials <= 0:
        parser.error(
            "--trials duhet te jete pozitiv."
        )

    # ------------------------------------------------------
    # ZGJEDHJA E BURIMIT
    # ------------------------------------------------------

    if SOURCE.exists():
        source_path = SOURCE

    elif FALLBACK_SOURCE.exists():
        source_path = FALLBACK_SOURCE

    else:
        raise FileNotFoundError(
            "Nuk u gjet submission-i V3."
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    original_document = read_json(
        source_path
    )

    lower, upper = [
        np.asarray(
            values,
            dtype=float
        )
        for values in udp.get_bounds()
    ]

    rng = np.random.default_rng(
        args.seed
    )

    attempts = [0, 0, 0, 0]
    successes = [0, 0, 0, 0]

    start_trial = 0

    # ------------------------------------------------------
    # RESUME
    # ------------------------------------------------------

    if args.resume:

        if not STATE.exists():
            raise FileNotFoundError(
                "Nuk ekziston checkpoint per --resume."
            )

        state = read_json(
            STATE
        )

        population = state["population"]

        start_trial = int(
            state["trial"]
        )

        attempts = state["attempts"]
        successes = state["successes"]

        rng.bit_generator.state = (
            state["rng_state"]
        )

        print(
            f"RESUME nga Trial {start_trial}",
            flush=True
        )

    # ------------------------------------------------------
    # START NGA V3
    # ------------------------------------------------------

    else:

        if STATE.exists():
            raise RuntimeError(
                "Ekziston nje V4 i meparshem. "
                "Per te vazhduar perdor --resume. "
                "Nuk po mbishkruajme rezultatet."
            )

        vectors = get_payload(
            original_document
        )["decisionVector"]

        if len(vectors) != MAX_POP:
            raise ValueError(
                "Submission V3 duhet te kete "
                "100 konfigurime."
            )

        population = []

        print(
            "Duke verifikuar konfigurimet V3...",
            flush=True
        )

        for i, vector in enumerate(
            vectors,
            start=1
        ):

            candidate = evaluate(
                vector
            )

            if candidate is None:
                raise RuntimeError(
                    f"Konfigurimi {i} eshte i pavlefshem."
                )

            population.append(
                candidate
            )

            if i % 25 == 0:
                print(
                    f"Verifikuar {i}/{len(vectors)}",
                    flush=True
                )

        population = reduce_population(
            population
        )

    # ------------------------------------------------------
    # VERIFIKIMI I POPULLATES
    # ------------------------------------------------------

    if len(population) < 3:
        raise RuntimeError(
            "Nuk ka mjaft konfigurime te vlefshme."
        )

    for i, item in enumerate(
        population,
        start=1
    ):

        actual = evaluate(
            item["x"]
        )

        if actual is None:
            raise RuntimeError(
                f"Konfigurimi {i} nuk eshte i vlefshem."
            )

        if not np.allclose(
            actual["f"],
            item["f"],
            rtol=0,
            atol=1e-8
        ):
            raise RuntimeError(
                f"Fitness nuk perputhet: {i}"
            )

    initial_hv = hypervolume(
        population
    )

    best_hv = initial_hv

    print(
        f"HV FILLESTAR: {initial_hv:.9f}",
        flush=True
    )

    print(
        f"Konfigurime: {len(population)}",
        flush=True
    )

    # V4 nuk duhet te niset nga rezultat me i
    # dobet se V3 qe kemi verifikuar.

    if (
        not args.resume
        and initial_hv < EXPECTED_MIN_HV - 1e-7
    ):
        raise RuntimeError(
            "Rezultati fillestar eshte me i dobet "
            "se rekordi V3. Kontrollo submission."
        )

    # ------------------------------------------------------
    # RUAJTJA FILLESTARE
    # ------------------------------------------------------

    save_checkpoint(
        original_document,
        population,
        start_trial,
        attempts,
        successes,
        rng
    )

    log_is_new = not LOG.exists()

    if args.resume:
        log_mode = "a"

    else:
        log_mode = "w"

    log_file = LOG.open(
        log_mode,
        newline="",
        encoding="utf-8"
    )

    writer = csv.writer(
        log_file
    )

    if log_is_new or not args.resume:
        writer.writerow([
            "trial",
            "operator",
            "valid",
            "accepted",
            "hypervolume",
            "J1",
            "J2"
        ])
        log_file.flush()

    total_trial = (
        start_trial + args.trials
    )

    last_trial = start_trial

    # ------------------------------------------------------
    # OPTIMIZIMI
    # ------------------------------------------------------

    try:

        for trial in range(
            start_trial + 1,
            total_trial + 1
        ):

            # Zgjedhim operatorin ne menyre adaptive
            # sipas suksesit te kaluar.

            weights = np.asarray([
                (successes[i] + 2)
                / (attempts[i] + 8)
                for i in range(4)
            ], dtype=float)

            probabilities = (
                weights / weights.sum()
            )

            mode = int(
                rng.choice(
                    4,
                    p=probabilities
                )
            )

            attempts[mode] += 1

            child = generate_candidate(
                population,
                mode,
                rng,
                lower,
                upper
            )

            candidate = evaluate(
                child
            )

            accepted = False

            if candidate is not None:

                proposed = reduce_population(
                    population + [candidate]
                )

                new_hv = hypervolume(
                    proposed
                )

                if new_hv > best_hv + 1e-10:

                    population = proposed
                    best_hv = new_hv

                    successes[mode] += 1
                    accepted = True

                    print(
                        f"PERMIRESIM Trial {trial}: "
                        f"HV={best_hv:.9f} | "
                        f"Mode={OPERATOR_NAMES[mode]}",
                        flush=True
                    )

            writer.writerow([
                trial,
                OPERATOR_NAMES[mode],
                candidate is not None,
                accepted,
                f"{best_hv:.12f}",
                (
                    candidate["f"][0]
                    if candidate is not None
                    else ""
                ),
                (
                    candidate["f"][1]
                    if candidate is not None
                    else ""
                )
            ])

            log_file.flush()

            last_trial = trial

            # Ruajme checkpoint çdo 25 prova
            # dhe gjithmone kur ka permiresim.

            if accepted or trial % 25 == 0:

                save_checkpoint(
                    original_document,
                    population,
                    trial,
                    attempts,
                    successes,
                    rng
                )

            if trial % 25 == 0:

                print(
                    f"Trial {trial}/{total_trial} | "
                    f"HV={best_hv:.9f} | "
                    f"Permiresime={sum(successes)}",
                    flush=True
                )

    except KeyboardInterrupt:

        print(
            "Optimizimi u nderpre. "
            "Po ruajme rezultatet.",
            flush=True
        )

    finally:

        save_checkpoint(
            original_document,
            population,
            last_trial,
            attempts,
            successes,
            rng
        )

        log_file.close()

        print(
            "\n========== REZULTATI V4 ==========",
            flush=True
        )

        print(
            f"HV fillestar: {initial_hv:.9f}",
            flush=True
        )

        print(
            f"HV final:     {best_hv:.9f}",
            flush=True
        )

        print(
            f"Permiresimi: "
            f"{best_hv - initial_hv:+.9f}",
            flush=True
        )

        print(
            f"Provat totale: {last_trial}",
            flush=True
        )

        print(
            f"Submission: {BEST}",
            flush=True
        )

        print(
            f"Checkpoint: {STATE}",
            flush=True
        )

        print(
            f"Historiku: {LOG}",
            flush=True
        )


# ==========================================================
# 14. NISJA E PROGRAMIT
# ==========================================================

if __name__ == "__main__":
    main()
