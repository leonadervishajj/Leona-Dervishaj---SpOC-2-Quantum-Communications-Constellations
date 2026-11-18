
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

from experiments.esa_numba_geometry import (
    fast_build_graph
)

from experiments.fcmaes_fast_fitness import (
    fast_average_shortest_path
)


REF = np.array([1.2, 1.4], dtype=float)

BAD = np.array(
    [9.0, 9.0, 1e6, 1e6],
    dtype=float
)


# ==========================================================
# HYPERVOLUME
# ==========================================================

def hypervolume(rows):

    points = []

    for row in rows:

        f = np.asarray(
            row["f"][:2],
            dtype=float
        )

        if (
            np.isfinite(f).all()
            and np.all(f < REF)
        ):
            points.append(f.tolist())

    if not points:
        return 0.0

    return float(
        pg.hypervolume(points).compute(REF)
    )


def esa_raw_score(rows):

    return -float(
        combine_scores(
            [r["f"] for r in rows]
        )
    )


def select_best(rows, target=100):

    selected = list(rows)

    while len(selected) > target:

        outside = [
            i
            for i, row in enumerate(selected)
            if (
                not np.isfinite(row["f"][:2]).all()
                or np.any(
                    np.asarray(row["f"][:2]) >= REF
                )
            )
        ]

        if outside:
            selected.pop(outside[0])
            continue

        points = [
            row["f"][:2]
            for row in selected
        ]

        index = int(
            pg.hypervolume(
                points
            ).least_contributor(REF)
        )

        selected.pop(index)

    return selected


# ==========================================================
# FILE HANDLING
# ==========================================================

def save_json(path, data):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp = path.with_name(
        path.name + ".tmp"
    )

    temp.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    os.replace(temp, path)


def vector_key(x):

    return tuple(
        np.round(
            np.asarray(x, dtype=float),
            10
        )
    )


# ==========================================================
# PARETO ARCHIVE
# ==========================================================

def update_archive(archive, row):

    f = np.asarray(
        row["f"][:2],
        dtype=float
    )

    remaining = []

    for old in archive:

        old_f = np.asarray(
            old["f"][:2],
            dtype=float
        )

        if np.all(old_f <= f):
            return archive

        if not np.all(f <= old_f):
            remaining.append(old)

    remaining.append(row)

    return remaining


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--seed-checkpoint",
        type=Path,
        default=None
    )

    parser.add_argument(
        "--evaluations",
        type=int,
        default=640
    )

    parser.add_argument(
        "--population",
        type=int,
        default=64
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026
    )

    parser.add_argument(
        "--no-fast",
        action="store_true"
    )

    args = parser.parse_args()

    if not args.base.is_file():
        parser.error("Baseline file nuk ekziston.")

    if (
        args.out.exists()
        and any(args.out.iterdir())
    ):
        parser.error(
            "Output folder nuk eshte bosh. "
            "Perdor nje folder te ri."
        )

    if (
        args.population < 16
        or args.population % 2 != 0
    ):
        parser.error(
            "Population duhet te jete cift >= 16."
        )

    if args.evaluations < args.population:
        parser.error(
            "Evaluations duhet >= population."
        )

    rng = np.random.default_rng(args.seed)

    lower, upper = [
        np.asarray(v, dtype=float)
        for v in udp.get_bounds()
    ]

    if len(lower) != 20:
        raise RuntimeError(
            "ESA duhet te kete 20 variabla."
        )

    source = json.loads(
        args.base.read_text(encoding="utf-8")
    )

    payload = (
        source[0]
        if isinstance(source, list)
        else source
    )

    vectors = payload["decisionVector"]

    if len(vectors) != 100:
        raise RuntimeError(
            "Baseline duhet te kete 100 vektore."
        )

    # ======================================================
    # DECISION VECTOR
    # ======================================================

    def repair(x):

        z = np.asarray(
            x,
            dtype=float
        ).copy()

        if (
            z.shape != (20,)
            or not np.isfinite(z).all()
        ):
            raise ValueError(
                "Decision vector i pavlefshem."
            )

        z = np.clip(z, lower, upper)

        # S/P/F and rovers are integer variables.
        z[10:] = np.rint(z[10:])

        return z

    # ======================================================
    # ROVER VALIDATION
    # ======================================================

    def rover_ok(x):

        ids = np.asarray(
            x[16:20],
            dtype=int
        )

        if len(set(ids.tolist())) != 4:
            return False

        constraint, _ = (
            udp.get_rover_constraint(
                udp.lambdas[ids],
                udp.phis[ids]
            )
        )

        return bool(
            np.isfinite(constraint)
            and constraint <= 0
        )

    # ======================================================
    # ESA FITNESS
    # ======================================================

    def fitness(x):

        f = np.asarray(
            udp.fitness(x.tolist()),
            dtype=float
        )

        if (
            f.shape != (4,)
            or not np.isfinite(f).all()
        ):
            return BAD.copy()

        return f

    # ======================================================
    # BASELINE VERIFICATION - ORIGINAL ESA
    # ======================================================

    print(
        "BASELINE: Verifikimi origjinal ESA 100/100...",
        flush=True
    )

    best = []

    for i, x in enumerate(vectors):

        z = repair(x)
        f = fitness(z)

        if (
            not np.isfinite(f).all()
            or np.any(f[2:] > 0)
        ):
            raise RuntimeError(
                f"Baseline i pavlefshem: {i}"
            )

        best.append({
            "x": z.tolist(),
            "f": f.tolist()
        })

    baseline_hv = hypervolume(best)

    baseline_raw = esa_raw_score(best)

    if baseline_hv <= 0:
        raise RuntimeError(
            "Baseline HV nuk eshte pozitiv."
        )

    # Detect score scaling used by local ESA version.
    scale = baseline_raw / baseline_hv

    if np.isclose(
        scale, 10000.0, rtol=1e-7, atol=1e-7
    ):
        scale = 10000.0

    elif np.isclose(
        scale, 1.0, rtol=1e-7, atol=1e-7
    ):
        scale = 1.0

    else:
        raise RuntimeError(
            "ESA score scale e panjohur: "
            f"{scale}. Kontrollo combine_scores."
        )

    normalized_esa = baseline_raw / scale

    if abs(
        normalized_esa - baseline_hv
    ) > 1e-8:
        raise RuntimeError(
            "ESA score nuk perputhet me PyGMO."
        )

    print(
        f"BASELINE VERIFIED 100/100 | "
        f"HV={baseline_hv:.12f}",
        flush=True
    )

    print(
        f"ESA RAW SCORE={baseline_raw:.12f} | "
        f"SCALE={scale:g}",
        flush=True
    )

    # ======================================================
    # CACHE + ARCHIVE
    # ======================================================

    cache = {
        vector_key(row["x"]): np.asarray(
            row["f"], dtype=float
        )
        for row in best
    }

    archive = []

    for row in best:

        if np.all(
            np.asarray(row["f"][:2]) < REF
        ):
            archive = update_archive(
                archive,
                row
            )

    current_hv = baseline_hv

    accepted = 0
    evaluated = 0
    invalid = 0
    errors = 0
    duplicates = 0

    args.out.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================================
    # CHECKPOINT
    # ======================================================

    def checkpoint():

        save_json(
            args.out / "checkpoint.json",
            {
                "hv_unverified": current_hv,
                "baseline_hv": baseline_hv,
                "score_scale": scale,
                "evaluated": evaluated,
                "invalid": invalid,
                "errors": errors,
                "duplicates": duplicates,
                "accepted": accepted,
                "population": best,
                "archive": archive
            }
        )

    # ======================================================
    # ACCEPT BETTER SOLUTIONS
    # ======================================================

    def consider(row):

        nonlocal archive
        nonlocal best
        nonlocal current_hv
        nonlocal accepted

        f = np.asarray(
            row["f"][:2],
            dtype=float
        )

        if (
            not np.isfinite(f).all()
            or np.any(f >= REF)
        ):
            return

        archive = update_archive(
            archive,
            row
        )

        if len(archive) > 300:
            archive = select_best(
                archive,
                250
            )

        candidate = select_best(
            best + [row],
            100
        )

        candidate_hv = hypervolume(
            candidate
        )

        if candidate_hv > current_hv + 1e-12:

            best = candidate
            current_hv = candidate_hv
            accepted += 1

            print(
                f"IMPROVED #{accepted} | "
                f"HV={current_hv:.12f} | "
                f"GAIN={current_hv-baseline_hv:.12f}",
                flush=True
            )

            checkpoint()

    # ======================================================
    # IMPORT PREVIOUS CHECKPOINT
    # Uses ORIGINAL ESA before enabling fast methods.
    # ======================================================

    if args.seed_checkpoint is not None:

        if not args.seed_checkpoint.is_file():
            parser.error(
                "Seed checkpoint nuk ekziston."
            )

        previous = json.loads(
            args.seed_checkpoint.read_text(
                encoding="utf-8"
            )
        )

        candidates = (
            previous.get("archive", [])
            + previous.get("population", [])
        )

        print(
            f"Importing {len(candidates)} candidates...",
            flush=True
        )

        for item in candidates:

            if (
                not isinstance(item, dict)
                or "x" not in item
            ):
                continue

            try:

                x = repair(item["x"])

                k = vector_key(x)

                if k in cache:
                    continue

                if not rover_ok(x):
                    continue

                f = fitness(x)

                if (
                    not np.isfinite(f).all()
                    or np.any(f[2:] > 0)
                ):
                    continue

                cache[k] = f.copy()

                consider({
                    "x": x.tolist(),
                    "f": f.tolist()
                })

            except (
                TypeError,
                ValueError,
                IndexError
            ):
                continue

    checkpoint()

    # ======================================================
    # PARENT SELECTION
    # ======================================================

    def parent():

        pool = (
            archive
            if archive and rng.random() < 0.7
            else best
        )

        index = int(
            rng.integers(len(pool))
        )

        return np.asarray(
            pool[index]["x"],
            dtype=float
        )

    # ======================================================
    # HYBRID MUTATION
    # ======================================================

    def mutate(x):

        z = repair(x)

        kind = int(
            rng.integers(6)
        )

        # DIFFERENTIAL EVOLUTION
        if kind == 0:

            a = parent()
            b = parent()

            mask = rng.random(10) < 0.5

            mask[
                int(rng.integers(10))
            ] = True

            z[:10] += (
                mask
                * rng.uniform(0.3, 1.0)
                * (a[:10] - b[:10])
            )

        # ORBITAL GEOMETRY
        elif kind == 1:

            js = rng.choice(
                10,
                size=int(rng.integers(1, 8)),
                replace=False
            )

            z[js] += (
                rng.normal(
                    0,
                    rng.choice([0.01, 0.05, 0.15]),
                    len(js)
                )
                * (upper[js] - lower[js])
            )

        # WALKER S/P/F
        elif kind == 2:

            js = rng.choice(
                np.arange(10, 16),
                size=int(rng.integers(1, 4)),
                replace=False
            )

            for j in js:

                if rng.random() < 0.7:
                    z[j] += rng.choice(
                        [-2, -1, 1, 2]
                    )
                else:
                    z[j] = rng.integers(
                        int(lower[j]),
                        int(upper[j]) + 1
                    )

        # ROVER SEARCH
        elif kind == 3:

            for _ in range(30):

                attempt = z.copy()

                js = rng.choice(
                    np.arange(16, 20),
                    size=int(rng.integers(1, 3)),
                    replace=False
                )

                for j in js:
                    attempt[j] = rng.integers(
                        int(lower[j]),
                        int(upper[j]) + 1
                    )

                attempt = repair(attempt)

                if rover_ok(attempt):
                    z = attempt
                    break

        # WALKER CROSSOVER
        elif kind == 4:

            other = parent()

            side = int(rng.integers(2))

            z[
                5 * side:5 * side + 5
            ] = other[
                5 * side:5 * side + 5
            ]

            z[
                10 + 3 * side:13 + 3 * side
            ] = other[
                10 + 3 * side:13 + 3 * side
            ]

        # GLOBAL RESTART
        else:

            z = parent()

            z[:10] += (
                rng.normal(0, 0.15, 10)
                * (upper[:10] - lower[:10])
            )

            js = rng.choice(
                np.arange(10, 16),
                size=3,
                replace=False
            )

            for j in js:
                z[j] = rng.integers(
                    int(lower[j]),
                    int(upper[j]) + 1
                )

        return repair(z)

    # ======================================================
    # FAST FITNESS
    # ======================================================

    def evaluate(x):

        nonlocal evaluated
        nonlocal invalid
        nonlocal errors
        nonlocal duplicates

        try:

            z = repair(x)
            k = vector_key(z)

            if k in cache:
                duplicates += 1
                return cache[k].copy()

            if not rover_ok(z):
                invalid += 1
                return BAD.copy()

            f = fitness(z)

            evaluated += 1

            if (
                not np.isfinite(f).all()
                or f.shape != (4,)
            ):
                errors += 1
                return BAD.copy()

            cache[k] = f.copy()

            if np.any(f[2:] > 0):
                invalid += 1
                return f

            consider({
                "x": z.tolist(),
                "f": f.tolist()
            })

            return f

        except Exception as exc:

            errors += 1

            if errors <= 3:
                print(
                    f"Fitness warning: {exc}",
                    flush=True
                )

            return BAD.copy()

    # ======================================================
    # FAST ESA ENGINE: NUMBA + IGRAPH
    # ======================================================

    original_graph = udp.build_graph
    original_paths = udp.average_shortest_path

    try:

        if not args.no_fast:

            udp.build_graph = MethodType(
                fast_build_graph,
                udp
            )

            udp.average_shortest_path = MethodType(
                fast_average_shortest_path,
                udp
            )

            print(
                "FAST ESA ENGINE: NUMBA + IGRAPH ON",
                flush=True
            )

        else:

            print(
                "FAST ESA ENGINE: ORIGINAL",
                flush=True
            )

        # ==================================================
        # FCMAES MODE
        # ==================================================

        def modifier(x):

            z = repair(x)

            if rng.random() < 0.8:
                return mutate(z)

            return z

        optimizer = mode.MODE(
            2,
            2,
            Bounds(lower, upper),
            popsize=args.population,
            nsga_update=False,
            modifier=modifier,
            rg=rng
        )

        starting = [
            best[int(i)]
            for i in rng.choice(
                len(best),
                args.population,
                replace=False
            )
        ]

        optimizer.set_guess(
            (
                np.asarray(
                    [r["x"] for r in starting],
                    dtype=float
                ),
                np.asarray(
                    [r["f"] for r in starting],
                    dtype=float
                )
            ),
            evaluate,
            rng
        )

        generations = (
            args.evaluations
            // args.population
        )

        for generation in range(
            1,
            generations + 1
        ):

            xs = optimizer.ask().copy()

            ys = np.asarray(
                [evaluate(x) for x in xs],
                dtype=float
            )

            optimizer.tell(ys)

            # Additional local search
            if generation % 2 == 0:

                for _ in range(8):
                    evaluate(
                        mutate(parent())
                    )

            checkpoint()

            print(
                f"Generation "
                f"{generation}/{generations} | "
                f"HV={current_hv:.12f} | "
                f"accepted={accepted} | "
                f"evaluated={evaluated} | "
                f"invalid={invalid} | "
                f"archive={len(archive)}",
                flush=True
            )

    finally:

        # Restore original ESA methods.
        udp.build_graph = original_graph

        udp.average_shortest_path = original_paths

    # ======================================================
    # FINAL VERIFICATION WITH ORIGINAL ESA
    # ======================================================

    print(
        "FINAL ESA VERIFICATION 100/100...",
        flush=True
    )

    verified = []

    for i, row in enumerate(best):

        x = repair(row["x"])

        f = fitness(x)

        if (
            not np.isfinite(f).all()
            or np.any(f[2:] > 0)
        ):
            raise RuntimeError(
                f"ESA verification failed: {i}"
            )

        verified.append({
            "x": x.tolist(),
            "f": f.tolist()
        })

    final_hv = hypervolume(verified)

    final_esa_hv = (
        esa_raw_score(verified) / scale
    )

    if (
        abs(final_hv - final_esa_hv) > 1e-8
        or abs(final_hv - current_hv) > 1e-8
        or final_hv < baseline_hv - 1e-10
    ):
        raise RuntimeError(
            "FINAL ESA MISMATCH! "
            "Submission nuk ruhet. "
            f"PyGMO={final_hv:.12f}, "
            f"ESA={final_esa_hv:.12f}, "
            f"FAST={current_hv:.12f}"
        )

    # ======================================================
    # SAVE VERIFIED SUBMISSION
    # ======================================================

    output_document = copy.deepcopy(source)

    target = (
        output_document[0]
        if isinstance(output_document, list)
        else output_document
    )

    target["decisionVector"] = [
        r["x"]
        for r in verified
    ]

    output = (
        args.out
        / "esa_submission_hybrid_best.json"
    )

    save_json(
        output,
        output_document
    )

    save_json(
        args.out / "verified_result.json",
        {
            "verified_hv": final_hv,
            "baseline_hv": baseline_hv,
            "gain": final_hv - baseline_hv,
            "score_scale": scale,
            "accepted": accepted,
            "evaluated": evaluated,
            "invalid": invalid,
            "errors": errors
        }
    )

    print(
        f"VERIFIED ESA 100/100 | "
        f"HV={final_hv:.12f} | "
        f"GAIN={final_hv-baseline_hv:.12f} | "
        f"accepted={accepted} | "
        f"errors={errors}",
        flush=True
    )

    print(
        f"FILE: {output}",
        flush=True
    )


if __name__ == "__main__":
    main()
