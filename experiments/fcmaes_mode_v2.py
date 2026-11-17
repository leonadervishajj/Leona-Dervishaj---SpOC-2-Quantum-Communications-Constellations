
"""ESA SpOC 2 - FCMAES MODE V2.

MODE + mixed integer mutations + Pareto archive.
Original ESA fitness is used for final verification.

Run from project root.
"""

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
from constellation import udp, combine_scores

from experiments.fcmaes_fast_fitness import (
    fast_average_shortest_path
)


REF = np.array([1.2, 1.4], dtype=float)
BAD = np.array([9.0, 9.0, 1e6, 1e6])


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_name(path.name + ".tmp")

    temp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    os.replace(temp, path)


def calc_hv(rows):
    if not rows:
        return 0.0

    points = np.array(
        [row["f"][:2] for row in rows],
        dtype=float
    )

    return float(
        pg.hypervolume(points).compute(REF)
    )


def select_best(rows, size=100):
    selected = list(rows)

    while len(selected) > size:
        points = np.array(
            [row["f"][:2] for row in selected],
            dtype=float
        )

        remove = int(
            pg.hypervolume(points).least_contributor(REF)
        )

        selected.pop(remove)

    return selected


def dominates(a, b):
    return (
        np.all(a <= b)
        and np.any(a < b)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base", type=Path, required=True
    )
    parser.add_argument(
        "--out", type=Path, required=True
    )
    parser.add_argument(
        "--evaluations", type=int, default=500
    )
    parser.add_argument(
        "--population", type=int, default=100
    )
    parser.add_argument(
        "--seed", type=int, default=2026
    )
    parser.add_argument(
        "--no-fast", action="store_true"
    )

    args = parser.parse_args()

    if args.population < 100:
        parser.error(
            "Per kete version perdor population >= 100."
        )

    if args.population % 2:
        parser.error(
            "Population duhet te jete numer cift."
        )

    if args.evaluations < args.population:
        parser.error(
            "Evaluations duhet te jete >= population."
        )

    if not args.base.is_file():
        parser.error("Baseline file nuk ekziston.")

    if args.out.exists() and any(args.out.iterdir()):
        parser.error(
            "Output folder nuk eshte bosh. "
            "Perdor nje folder te ri."
        )

    rng = np.random.default_rng(args.seed)

    lb, ub = [
        np.asarray(b, dtype=float)
        for b in udp.get_bounds()
    ]

    if len(lb) != 20 or len(ub) != 20:
        raise RuntimeError("Priten 20 parametra.")

    def repair(x):
        x = np.asarray(x, dtype=float).copy()
        x = np.clip(x, lb, ub)

        x[10:] = np.rint(x[10:])
        return x

    def rover_ok(x):
        ids = np.asarray(
            repair(x)[16:20],
            dtype=int
        )

        if len(set(ids.tolist())) != 4:
            return False

        checker = getattr(
            udp, "get_rover_constraint", None
        )
        lambdas = getattr(udp, "lambdas", None)
        phis = getattr(udp, "phis", None)

        if (
            checker is None
            or lambdas is None
            or phis is None
        ):
            return True

        try:
            result = checker(
                np.asarray(lambdas)[ids],
                np.asarray(phis)[ids]
            )

            constraint = (
                result[0]
                if isinstance(result, (tuple, list))
                else result
            )

            return bool(
                np.all(np.asarray(constraint) <= 0)
            )

        except (
            TypeError, ValueError,
            IndexError, AttributeError
        ):
            # Uncertain helper: ESA fitness decides.
            return True

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
        raise RuntimeError(
            "Baseline duhet te kete 100 vektore."
        )

    print(
        "Verifikimi i bazes me modelin ESA...",
        flush=True
    )

    best = []

    for i, x in enumerate(vectors):
        x = repair(x)

        f = np.asarray(
            udp.fitness(x.tolist()),
            dtype=float
        )

        if (
            f.shape != (4,)
            or not np.isfinite(f).all()
            or np.any(f[2:] > 0)
        ):
            raise RuntimeError(
                f"Baseline i pavlefshem: {i}"
            )

        best.append({
            "x": x.tolist(),
            "f": f.tolist()
        })

    initial_hv = calc_hv(best)

    esa_hv = -float(
        combine_scores(
            [r["f"] for r in best]
        )
    ) / 10000.0

    if abs(initial_hv - esa_hv) > 1e-8:
        raise RuntimeError(
            "HV nuk perputhet me ESA."
        )

    best_hv = initial_hv

    print(
        f"BASELINE ESA VERIFIED 100/100 | "
        f"HV={best_hv:.12f}",
        flush=True
    )

    args.out.mkdir(parents=True, exist_ok=True)

    output = (
        args.out / "esa_submission_mode_v2_best.json"
    )
    checkpoint = args.out / "checkpoint.json"

    # Pareto archive: retain useful solutions even
    # if they do not improve the 100-point HV yet.
    archive = []

    def archive_add(row):
        nonlocal archive

        y = np.asarray(row["f"][:2])

        for old in archive:
            old_y = np.asarray(old["f"][:2])

            if (
                np.array_equal(old_y, y)
                or dominates(old_y, y)
            ):
                return False

        archive = [
            old for old in archive
            if not dominates(
                y,
                np.asarray(old["f"][:2])
            )
        ]

        archive.append(row)

        if len(archive) > 250:
            archive = select_best(archive, 250)

        return True

    for row in best:
        archive_add(row)

    seen = {
        tuple(np.round(r["x"], 10))
        for r in best
    }

    calls = 0
    valid = 0
    invalid = 0
    accepted = 0
    errors = 0

    def save():
        submission = copy.deepcopy(document)

        target = (
            submission[0]
            if isinstance(submission, list)
            else submission
        )

        target["decisionVector"] = [
            r["x"] for r in best
        ]

        save_json(output, submission)

        save_json(checkpoint, {
            "hv": best_hv,
            "initial_hv": initial_hv,
            "calls": calls,
            "valid": valid,
            "invalid": invalid,
            "accepted": accepted,
            "errors": errors,
            "population": best,
            "archive": archive
        })

    # Protected starting checkpoint.
    save()

    # Mutation strategy for mixed variables.
    def modifier(x):
        z = repair(x)

        # Occasional restart near an archive member.
        if archive and rng.random() < 0.20:
            parent = archive[
                int(rng.integers(len(archive)))
            ]

            base = np.asarray(
                parent["x"],
                dtype=float
            )

            span = ub[:10] - lb[:10]

            z[:10] = (
                base[:10]
                + rng.normal(
                    0, 0.05, size=10
                ) * span
            )

            z[10:] = base[10:]

        # Walker structure mutation.
        if rng.random() < 0.45:
            j = int(rng.integers(10, 16))

            if rng.random() < 0.75:
                z[j] += rng.choice([-1, 1])
            else:
                z[j] = rng.integers(
                    int(lb[j]),
                    int(ub[j]) + 1
                )

        # Rover mutation with feasibility precheck.
        if rng.random() < 0.30:
            original = repair(z)

            for _ in range(30):
                trial = original.copy()

                j = int(rng.integers(16, 20))

                trial[j] = rng.integers(
                    int(lb[j]),
                    int(ub[j]) + 1
                )

                trial = repair(trial)

                if rover_ok(trial):
                    z = trial
                    break

        return repair(z)

    # Evaluations use fast path; final check does not.
    original_method = udp.average_shortest_path

    def evaluate(x):
        nonlocal calls, valid, invalid
        nonlocal accepted, errors, best, best_hv

        z = repair(x)
        calls += 1

        if not rover_ok(z):
            invalid += 1
            return BAD.copy()

        key = tuple(np.round(z, 10))

        if key in seen:
            # MODE still needs the objective values.
            try:
                return np.asarray(
                    udp.fitness(z.tolist()),
                    dtype=float
                )
            except Exception:
                errors += 1
                return BAD.copy()

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

            return BAD.copy()

        if (
            f.shape != (4,)
            or not np.isfinite(f).all()
        ):
            errors += 1
            return BAD.copy()

        if np.any(f[2:] > 0):
            invalid += 1
            return f

        valid += 1

        if np.all(f[:2] < REF):
            row = {
                "x": z.tolist(),
                "f": f.tolist()
            }

            archive_add(row)

            candidate = select_best(
                best + [row], 100
            )

            new_hv = calc_hv(candidate)

            if new_hv > best_hv + 1e-12:
                best = candidate
                best_hv = new_hv
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
        if not args.no_fast:
            udp.average_shortest_path = MethodType(
                fast_average_shortest_path,
                udp
            )

            print(
                "FAST FITNESS: igraph ON",
                flush=True
            )

        optimizer = mode.MODE(
            2,
            2,
            Bounds(lb, ub),
            popsize=args.population,
            nsga_update=False,
            modifier=modifier,
            rg=rng
        )

        optimizer.set_guess(
            (
                np.asarray(
                    [r["x"] for r in best],
                    dtype=float
                ),
                np.asarray(
                    [r["f"] for r in best],
                    dtype=float
                )
            ),
            evaluate,
            rng
        )

        generations = (
            args.evaluations // args.population
        )

        for gen in range(
            1, generations + 1
        ):
            xs = optimizer.ask().copy()

            ys = np.asarray(
                [evaluate(x) for x in xs],
                dtype=float
            )

            optimizer.tell(ys)

            print(
                f"Generation {gen}/{generations} | "
                f"calls={calls} | "
                f"valid={valid} | "
                f"invalid={invalid} | "
                f"accepted={accepted} | "
                f"archive={len(archive)} | "
                f"HV={best_hv:.12f}",
                flush=True
            )

            save()

    finally:
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
                f"Verification failed at {i}"
            )

        checked.append({
            "x": row["x"],
            "f": f.tolist()
        })

    verified_hv = -float(
        combine_scores(
            [r["f"] for r in checked]
        )
    ) / 10000.0

    if (
        abs(verified_hv - best_hv) > 1e-8
        or verified_hv + 1e-10 < initial_hv
    ):
        raise RuntimeError(
            "Final HV mismatch with ESA."
        )

    best = checked
    best_hv = verified_hv
    save()

    print(
        f"VERIFIED ESA 100/100 | "
        f"HV={best_hv:.12f} | "
        f"GAIN={best_hv-initial_hv:.12f} | "
        f"accepted={accepted} | "
        f"errors={errors}",
        flush=True
    )

    print("FILE:", output, flush=True)


if __name__ == "__main__":
    main()
