
"""ESA SpOC 2 - Hypervolume-Guided Evolution V3."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pygmo as pg
from constellation import udp


REFERENCE = [1.2, 1.4]
SOURCE = Path("submission/esa_submission_hybrid_v3.json")
OUTDIR = Path("results/pareto_hv_v3")

BEST_PATH = OUTDIR / "esa_submission_best.json"
STATE_PATH = OUTDIR / "state.json"

MAX_SIZE = 100
DIM = 20
DISCRETE = np.arange(10, 20)


# 1. VLERESIMI ME MODELIN ESA

def evaluate(x):
    values = np.asarray(x, dtype=float)

    if values.shape != (DIM,):
        return None

    if not np.all(np.isfinite(values)):
        return None

    result = np.asarray(
        udp.fitness(values.tolist()),
        dtype=float
    )

    if result.size < 4:
        return None

    if not np.all(np.isfinite(result)):
        return None

    # Kufizimet duhet te jene <= 0
    if np.any(result[2:] > 0):
        return None

    return {
        "x": values.tolist(),
        "f": result[:2].tolist()
    }


# 2. LLOGARITJA E HYPERVOLUME

def hv(population):
    points = [
        item["f"]
        for item in population
        if item["f"][0] < REFERENCE[0]
        and item["f"][1] < REFERENCE[1]
    ]

    if not points:
        return 0.0

    return float(
        pg.hypervolume(points).compute(REFERENCE)
    )


# 3. LARGIMI I DUPLIKATEVE

def unique(population):
    seen = set()
    result = []

    for item in population:
        key = tuple(float(v) for v in item["x"])

        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


# 4. MBAJME MAKSIMUMI 100 KONFIGURIME

def reduce_to_limit(population, limit=MAX_SIZE):
    population = unique(population)

    while len(population) > limit:
        total_hv = hv(population)

        losses = []

        for i in range(len(population)):
            remaining = (
                population[:i] + population[i + 1:]
            )

            loss = total_hv - hv(remaining)
            losses.append(loss)

        remove_index = int(np.argmin(losses))
        population.pop(remove_index)

    return population


# 5. GJETJA E PIKAVE PARETO

def pareto_indices(population):
    result = []

    for i, a in enumerate(population):
        x, y = a["f"]

        if x >= REFERENCE[0] or y >= REFERENCE[1]:
            continue

        dominated = False

        for j, b in enumerate(population):
            if i == j:
                continue

            bx, by = b["f"]

            if (
                bx <= x
                and by <= y
                and (bx < x or by < y)
            ):
                dominated = True
                break

        if not dominated:
            result.append(i)

    return result


# 6. ZGJEDHJA E PRINDERVE NGA HAPESIRAT PARETO

def choose_gap_parents(population, rng):
    indices = pareto_indices(population)

    if len(indices) < 2:
        a = int(rng.integers(len(population)))
        b = int(rng.integers(len(population)))

        return population[a], population[b]

    indices.sort(
        key=lambda i: population[i]["f"][0]
    )

    points = np.asarray(
        [population[i]["f"] for i in indices],
        dtype=float
    )

    differences = np.abs(
        np.diff(points, axis=0)
    )

    widths = np.maximum(
        np.ptp(points, axis=0),
        1e-12
    )

    gaps = (
        differences[:, 0] / widths[0]
        + differences[:, 1] / widths[1]
        + 0.02
    )

    probabilities = gaps / gaps.sum()

    k = int(
        rng.choice(len(gaps), p=probabilities)
    )

    return (
        population[indices[k]],
        population[indices[k + 1]]
    )


# 7. KUFIJTE E PARAMETRAVE

def bounded(x, lower, upper):
    x = np.clip(
        np.asarray(x, dtype=float),
        lower,
        upper
    )

    # Parametrat diskrete rrumbullakohen
    x[DISCRETE] = np.rint(x[DISCRETE])

    return x


# 8. KRIJIMI I KANDIDATEVE TE RINJ

def propose(population, rng, lower, upper, iteration):
    span = upper - lower

    # Kater strategji te ndryshme
    mode = iteration % 4

    if mode == 0:
        # STRATEGJIA 1:
        # Crossover mes dy zgjidhjeve Pareto

        parent_a, parent_b = choose_gap_parents(
            population, rng
        )

        a = np.asarray(
            parent_a["x"], dtype=float
        )

        b = np.asarray(
            parent_b["x"], dtype=float
        )

        alpha = float(
            rng.uniform(0.15, 0.85)
        )

        child = (
            alpha * a
            + (1 - alpha) * b
        )

        switches = (
            rng.random(len(DISCRETE)) < 0.5
        )

        child[DISCRETE] = np.where(
            switches,
            a[DISCRETE],
            b[DISCRETE]
        )

        indices = rng.choice(
            10,
            size=int(rng.integers(1, 4)),
            replace=False
        )

        child[indices] += (
            rng.normal(
                0, 0.01, size=len(indices)
            ) * span[indices]
        )

    elif mode == 1:
        # STRATEGJIA 2:
        # Mutacion adaptiv lokal

        base = population[
            int(rng.integers(len(population)))
        ]

        child = np.asarray(
            base["x"], dtype=float
        ).copy()

        scale = float(
            rng.choice([
                0.001,
                0.003,
                0.01,
                0.03,
                0.08
            ])
        )

        indices = rng.choice(
            DIM,
            size=int(rng.integers(1, 6)),
            replace=False
        )

        child[indices] += (
            rng.normal(
                0, scale, size=len(indices)
            ) * span[indices]
        )

    elif mode == 2:
        # STRATEGJIA 3:
        # Differential Evolution

        indices = rng.choice(
            len(population),
            size=3,
            replace=False
        )

        base, a, b = [
            np.asarray(
                population[int(i)]["x"],
                dtype=float
            )
            for i in indices
        ]

        child = base.copy()

        mask = rng.random(DIM) < 0.5

        mask[
            int(rng.integers(DIM))
        ] = True

        factor = float(
            rng.uniform(0.3, 1.1)
        )

        child[mask] = (
            base[mask]
            + factor * (a[mask] - b[mask])
        )

    else:
        # STRATEGJIA 4:
        # Ndryshime diskrete dhe orbitale

        base = population[
            int(rng.integers(len(population)))
        ]

        child = np.asarray(
            base["x"], dtype=float
        ).copy()

        discrete_indices = rng.choice(
            DISCRETE,
            size=int(rng.integers(1, 5)),
            replace=False
        )

        for i in discrete_indices:
            i = int(i)

            if rng.random() < 0.18:
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

        continuous_indices = rng.choice(
            10,
            size=int(rng.integers(1, 4)),
            replace=False
        )

        child[continuous_indices] += (
            rng.normal(
                0,
                0.03,
                size=len(continuous_indices)
            ) * span[continuous_indices]
        )

    return (
        bounded(child, lower, upper),
        mode
    )


# 9. RUAJTJA E REZULTATEVE

def atomic_json(path, data):
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )

    temporary.replace(path)


def make_submission(template, population):
    if isinstance(template, list):
        result = dict(template[0])

        result["decisionVector"] = [
            item["x"]
            for item in population
        ]

        return [result]

    result = dict(template)

    result["decisionVector"] = [
        item["x"]
        for item in population
    ]

    return result


def save(template, population, score, iteration, counters):
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True
    )

    atomic_json(
        BEST_PATH,
        make_submission(template, population)
    )

    atomic_json(
        STATE_PATH,
        {
            "trial": iteration,
            "hypervolume": score,
            "population": population,
            "counters": counters
        }
    )


# 10. PROGRAMI KRYESOR

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trials",
        type=int,
        default=50
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
            "--trials must be positive"
        )

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Mungon submission-i: {SOURCE}"
        )

    template = json.loads(
        SOURCE.read_text(encoding="utf-8")
    )

    payload = (
        template[0]
        if isinstance(template, list)
        else template
    )

    vectors = payload["decisionVector"]

    if len(vectors) != MAX_SIZE:
        raise ValueError(
            "Priten 100 konfigurime."
        )

    if any(len(x) != DIM for x in vectors):
        raise ValueError(
            "Secili konfigurim duhet te kete 20 parametra."
        )

    lower, upper = [
        np.asarray(z, dtype=float)
        for z in udp.get_bounds()
    ]

    rng = np.random.default_rng(args.seed)

    start_trial = 0

    counters = {
        "valid": 0,
        "accepted": 0,
        "per_mode": [0, 0, 0, 0]
    }

    # VAZHDIMI NGA CHECKPOINT

    if args.resume and STATE_PATH.exists():
        state = json.loads(
            STATE_PATH.read_text(
                encoding="utf-8"
            )
        )

        population = state["population"]
        start_trial = int(state["trial"])
        counters = state["counters"]

        rng = np.random.default_rng(
            args.seed + start_trial
        )

        print(
            f"RESUME nga trial {start_trial}",
            flush=True
        )

    else:
        population = []

        print(
            "Duke verifikuar 100 konfigurimet Hybrid...",
            flush=True
        )

        for k, x in enumerate(vectors, start=1):
            candidate = evaluate(x)

            if candidate is None:
                raise RuntimeError(
                    f"Konfigurimi {k} nuk eshte i vlefshem!"
                )

            population.append(candidate)

        population = reduce_to_limit(population)

    # VERIFIKIMI I FITNESS-IT

    for i, item in enumerate(population):
        actual = evaluate(item["x"])

        if actual is None:
            raise RuntimeError(
                f"Konfigurimi {i} eshte i pavlefshem."
            )

        if not np.allclose(
            actual["f"],
            item["f"],
            atol=1e-8,
            rtol=0
        ):
            raise RuntimeError(
                f"Fitness nuk perputhet: {i}"
            )

    if len(population) < 3:
        raise RuntimeError(
            "Duhet te kemi te pakten 3 konfigurime."
        )

    start_hv = hv(population)
    best_hv = start_hv

    print(
        f"HV FILLESTAR: {start_hv:.9f} | "
        f"Arkivi: {len(population)}",
        flush=True
    )

    # SIGUROHEMI QE PO NISEMI NGA REKORDI YNE

    if (
        not args.resume
        and abs(start_hv - 0.610332025) > 1e-6
    ):
        raise RuntimeError(
            "HV fillestar nuk perputhet me rekordin Hybrid!"
        )

    save(
        template,
        population,
        best_hv,
        start_trial,
        counters
    )

    end_trial = start_trial + args.trials

    last_trial = start_trial

    try:
        for trial in range(
            start_trial + 1,
            end_trial + 1
        ):
            last_trial = trial

            child, mode = propose(
                population,
                rng,
                lower,
                upper,
                trial
            )

            candidate = evaluate(child)

            if candidate is not None:
                counters["valid"] += 1

                proposed = reduce_to_limit(
                    population + [candidate]
                )

                new_hv = hv(proposed)

                if new_hv > best_hv + 1e-10:
                    population = proposed
                    best_hv = new_hv

                    counters["accepted"] += 1
                    counters["per_mode"][mode] += 1

                    save(
                        template,
                        population,
                        best_hv,
                        trial,
                        counters
                    )

                    print(
                        f"PERMIRESIM Trial {trial}: "
                        f"HV={best_hv:.9f} | "
                        f"Mode={mode}",
                        flush=True
                    )

            if trial % 10 == 0:
                print(
                    f"Trial {trial}/{end_trial} | "
                    f"HV={best_hv:.9f} | "
                    f"Valid={counters['valid']} | "
                    f"Permiresime={counters['accepted']}",
                    flush=True
                )

            if trial % 25 == 0:
                save(
                    template,
                    population,
                    best_hv,
                    trial,
                    counters
                )

    except KeyboardInterrupt:
        print(
            "Testimi u nderpre. "
            "Rezultati me i mire do te ruhet.",
            flush=True
        )

    finally:
        save(
            template,
            population,
            best_hv,
            last_trial,
            counters
        )

        print(
            "\n========== REZULTATI V3 ==========",
            flush=True
        )

        print(
            f"Fillestar: {start_hv:.9f}",
            flush=True
        )

        print(
            f"Final:     {best_hv:.9f}",
            flush=True
        )

        print(
            f"Permiresimi: {best_hv - start_hv:+.9f}",
            flush=True
        )

        print(
            f"Fajlli: {BEST_PATH}",
            flush=True
        )


# KY RRESHT E NIS PROGRAMIN

if __name__ == "__main__":
    main()
