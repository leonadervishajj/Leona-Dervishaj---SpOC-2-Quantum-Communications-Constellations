
import argparse
import copy
import json
import os
from pathlib import Path
from types import MethodType

import numpy as np
import pygmo as pg

from scipy.optimize import Bounds
from fcmaes import mode
from constellation import udp

from experiments.fcmaes_fast_fitness import (
    fast_average_shortest_path
)

REF = np.array([1.2, 1.4], dtype=float)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            data, f, indent=2, ensure_ascii=False
        )

    os.replace(tmp, path)


def hv(rows):
    points = np.asarray(
        [r["f"][:2] for r in rows],
        dtype=float
    )

    return float(
        pg.hypervolume(points).compute(REF)
    )


def select100(rows):
    result = list(rows)

    while len(result) > 100:
        points = np.asarray(
            [r["f"][:2] for r in result],
            dtype=float
        )

        idx = int(
            pg.hypervolume(points).least_contributor(REF)
        )

        result.pop(idx)

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base", type=Path, required=True
    )
    parser.add_argument(
        "--out", type=Path, required=True
    )
    parser.add_argument(
        "--evaluations", type=int, default=240
    )
    parser.add_argument(
        "--population", type=int, default=40
    )
    parser.add_argument(
        "--seed", type=int, default=77
    )
    parser.add_argument(
        "--no-fast", action="store_true"
    )

    args = parser.parse_args()

    if (
        args.population < 8
        or args.evaluations < args.population
    ):
        parser.error(
            "population >= 8 dhe evaluations >= population"
        )

    if args.out.exists() and any(args.out.iterdir()):
        parser.error(
            "Folderi i rezultateve nuk eshte bosh. "
            "Perdor nje folder te ri."
        )

    if not args.base.is_file():
        parser.error("Nuk ekziston skedari baze.")

    document = json.loads(
        args.base.read_text(encoding="utf-8")
    )

    payload = (
        document[0]
        if isinstance(document, list)
        else document
    )

    vectors = payload["decisionVector"]

    if len(vectors) != 100:
        raise ValueError(
            "Submission duhet te kete 100 vektore."
        )

    lb, ub = [
        np.asarray(z, dtype=float)
        for z in udp.get_bounds()
    ]

    if len(lb) != 20:
        raise ValueError("Priten 20 parametra.")

    ub[10:] = np.floor(ub[10:])
    rng = np.random.default_rng(args.seed)

    def repair(x):
        z = np.clip(
            np.asarray(x, dtype=float),
            lb,
            ub
        ).copy()

        z[10:] = np.rint(z[10:])

        return z

    # Verify the original ESA baseline.
    baseline = []

    print(
        "Po verifikohen 100 konfigurimet e raundit 7...",
        flush=True
    )

    for i, x in enumerate(vectors):
        z = repair(x)

        f = np.asarray(
            udp.fitness(z.tolist()),
            dtype=float
        )

        if (
            f.shape != (4,)
            or not np.all(np.isfinite(f))
            or np.any(f[2:] > 0)
        ):
            raise RuntimeError(
                f"Konfigurimi {i} eshte i pavlefshem."
            )

        baseline.append({
            "x": z.tolist(),
            "f": f.tolist()
        })

    best = list(baseline)
    initial_hv = hv(best)
    best_hv = initial_hv

    print(
        f"BASELINE ESA VERIFIED 100/100 | "
        f"HV={best_hv:.12f}",
        flush=True
    )

    args.out.mkdir(
        parents=True,
        exist_ok=True
    )

    final_file = (
        args.out / "esa_submission_mode_best.json"
    )

    candidate_file = (
        args.out / "candidate_unverified.json"
    )

    checkpoint_file = (
        args.out / "checkpoint.json"
    )

    seen = {
        tuple(np.round(r["x"], 10))
        for r in best
    }

    calls = 0
    accepted = 0
    valid = 0
    invalid = 0
    errors = 0

    def save():
        result = copy.deepcopy(document)

        obj = (
            result[0]
            if isinstance(result, list)
            else result
        )

        obj["decisionVector"] = [
            r["x"] for r in best
        ]

        atomic_json(candidate_file, result)

        atomic_json(checkpoint_file, {
            "HV": best_hv,
            "baseline_HV": initial_hv,
            "evaluations": calls,
            "accepted": accepted,
            "valid": valid,
            "invalid": invalid,
            "errors": errors,
            "population": best
        })

    save()

    original_method = udp.average_shortest_path

    if not args.no_fast:
        udp.average_shortest_path = MethodType(
            fast_average_shortest_path,
            udp
        )

        print(
            "FAST FITNESS: igraph aktiv.",
            flush=True
        )

    # Mixed integer mutation for Walker and rover genes.
    def modifier(x):
        z = repair(x)

        if rng.random() < 0.55:
            j = int(rng.integers(10, 16))

            z[j] = np.clip(
                z[j] + rng.choice([-1, 1]),
                lb[j],
                ub[j]
            )

        if rng.random() < 0.25:
            j = int(rng.integers(16, 20))

            z[j] = rng.integers(
                int(lb[j]),
                int(ub[j]) + 1
            )

        return z

    def evaluate(x):
        nonlocal calls, accepted, valid
        nonlocal invalid, errors, best, best_hv

        z = repair(x)
        calls += 1

        key = tuple(np.round(z, 10))

        if key in seen:
            try:
                return np.asarray(
                    udp.fitness(z.tolist()),
                    dtype=float
                )
            except Exception:
                return np.array(
                    [9.0, 9.0, 1e6, 1e6]
                )

        seen.add(key)

        try:
            f = np.asarray(
                udp.fitness(z.tolist()),
                dtype=float
            )

        except Exception as exc:
            errors += 1

            if errors <= 3:
                print(
                    "Fitness error:",
                    repr(exc),
                    flush=True
                )

            return np.array(
                [9.0, 9.0, 1e6, 1e6]
            )

        if (
            f.shape != (4,)
            or not np.isfinite(f).all()
        ):
            errors += 1

            return np.array(
                [9.0, 9.0, 1e6, 1e6]
            )

        if np.any(f[2:] > 0):
            invalid += 1
            return f

        valid += 1

        if np.all(f[:2] < REF):
            candidate = {
                "x": z.tolist(),
                "f": f.tolist()
            }

            updated = select100(
                best + [candidate]
            )

            next_hv = hv(updated)

            if next_hv > best_hv + 1e-12:
                best = updated
                best_hv = next_hv
                accepted += 1

                print(
                    f"IMPROVED #{accepted} | "
                    f"HV={best_hv:.12f} | "
                    f"step={calls}",
                    flush=True
                )

                save()

        return f

    try:
        solver = mode.MODE(
            2,
            2,
            Bounds(lb, ub),
            popsize=args.population,
            nsga_update=False,
            modifier=modifier,
            rg=rng
        )

        # Start with the verified Pareto population.
        initial_x = np.asarray(
            [r["x"] for r in baseline],
            dtype=float
        )

        initial_y = np.asarray(
            [r["f"] for r in baseline],
            dtype=float
        )

        solver.set_guess(
            (initial_x, initial_y),
            evaluate,
            rng
        )

        generations = (
            args.evaluations // args.population
        )

        for generation in range(
            1, generations + 1
        ):
            proposals = solver.ask().copy()

            results = np.asarray(
                [evaluate(x) for x in proposals],
                dtype=float
            )

            solver.tell(results)

            print(
                f"Generation {generation}/{generations} | "
                f"calls={calls} | "
                f"valid={valid} | "
                f"invalid={invalid} | "
                f"accepted={accepted} | "
                f"HV={best_hv:.12f}",
                flush=True
            )

            save()

    finally:
        # Restore original ESA fitness behavior.
        udp.average_shortest_path = original_method

    print(
        "FINAL ESA VERIFICATION...",
        flush=True
    )

    checked = []

    for i, row in enumerate(best):
        f = np.asarray(
            udp.fitness(row["x"]),
            dtype=float
        )

        if (
            f.shape != (4,)
            or not np.isfinite(f).all()
            or np.any(f[2:] > 0)
        ):
            raise RuntimeError(
                f"ESA verification failed at {i}"
            )

        checked.append({
            "x": row["x"],
            "f": f.tolist()
        })

    official_hv = hv(checked)

    if (
        abs(official_hv - best_hv) > 1e-8
        or official_hv + 1e-10 < initial_hv
    ):
        raise RuntimeError(
            f"HV mismatch: MODE={best_hv:.12f}, "
            f"ESA={official_hv:.12f}"
        )

    best = checked
    best_hv = official_hv

    save()

    atomic_json(
        final_file,
        json.loads(
            candidate_file.read_text(
                encoding="utf-8"
            )
        )
    )

    print(
        f"VERIFIED ESA 100/100 | "
        f"HV={best_hv:.12f} | "
        f"GAIN={best_hv-initial_hv:.12f} | "
        f"accepted={accepted} | errors={errors}",
        flush=True
    )

    print(
        "FILE:",
        final_file,
        flush=True
    )


if __name__ == "__main__":
    main()
