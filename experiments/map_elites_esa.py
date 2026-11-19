
"""ESA SpOC2 - Quality Diversity / MAP-Elites.

Explores different Walker structures and rover configurations.
Uses Numba + igraph for accelerated evaluation.
Does not modify the original constellation.py.
"""

import argparse
import copy
import json
import os

from pathlib import Path
from types import MethodType

import numpy as np
import pygmo as pg

from constellation import udp, combine_scores

from experiments.esa_numba_geometry import (
    fast_build_graph
)

from experiments.fcmaes_fast_fitness import (
    fast_average_shortest_path
)


REF = np.array([1.2, 1.4], dtype=float)


# ==========================================================
# HYPERVOLUME
# ==========================================================

def hv(rows):

    points = [
        row["f"][:2]
        for row in rows
        if np.all(
            np.asarray(row["f"][:2]) < REF
        )
    ]

    if not points:
        return 0.0

    return float(
        pg.hypervolume(points).compute(REF)
    )


def select100(rows):

    selected = list(rows)

    while len(selected) > 100:

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

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )

    os.replace(temp, path)


# ==========================================================
# MAP-ELITES NICHE
# ==========================================================

def niche(x):

    # Number of satellites in Walker 1
    n1 = int(x[10] * x[11])

    # Number of satellites in Walker 2
    n2 = int(x[13] * x[14])

    # Categories based on constellation structure.
    return (
        min(n1 // 12, 8),
        min(n2 // 12, 8),
        int(x[12]) // 3,
        int(x[15]) // 3
    )


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
        type=Path
    )

    parser.add_argument(
        "--evaluations",
        type=int,
        default=2000
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3030
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
            "Output folder nuk eshte bosh."
        )

    if args.evaluations < 1:
        parser.error(
            "Evaluations duhet >= 1."
        )

    rng = np.random.default_rng(
        args.seed
    )

    lower, upper = [
        np.asarray(v, dtype=float)
        for v in udp.get_bounds()
    ]

    document = json.loads(
        args.base.read_text(
            encoding="utf-8"
        )
    )

    payload = (
        document[0]
        if isinstance(document, list)
        else document
    )

    vectors = payload["decisionVector"]

    if len(vectors) != 100:
        raise RuntimeError(
            "Baseline kerkon 100 vektore."
        )

    # ======================================================
    # DECISION VECTOR
    # ======================================================

    def repair(x):

        z = np.asarray(
            x,
            dtype=float
        )

        if (
            z.shape != (20,)
            or not np.isfinite(z).all()
        ):
            raise ValueError(
                "Decision vector i pavlefshem."
            )

        z = np.clip(
            z,
            lower,
            upper
        ).copy()

        z[10:] = np.rint(
            z[10:]
        )

        return z

    # ======================================================
    # ROVER CONSTRAINT
    # ======================================================

    def rover_ok(x):

        ids = np.asarray(
            x[16:20],
            dtype=int
        )

        if len(set(ids.tolist())) < 4:
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
    # ORIGINAL ESA FITNESS
    # ======================================================

    def original_row(x):

        z = repair(x)

        f = np.asarray(
            udp.fitness(z.tolist()),
            dtype=float
        )

        if (
            f.shape != (4,)
            or not np.isfinite(f).all()
            or np.any(f[2:] > 0)
        ):
            raise ValueError(
                "ESA solution invalid."
            )

        return {
            "x": z.tolist(),
            "f": f.tolist()
        }

    # ======================================================
    # BASELINE ESA VERIFICATION
    # ======================================================

    print(
        "BASELINE: ORIGINAL ESA 100/100...",
        flush=True
    )

    best = [
        original_row(x)
        for x in vectors
    ]

    baseline_hv = hv(best)

    raw_score = -float(
        combine_scores(
            [row["f"] for row in best]
        )
    )

    scale = (
        raw_score / baseline_hv
        if baseline_hv > 0
        else float("nan")
    )

    if np.isclose(
        scale,
        10000.0,
        rtol=1e-7
    ):
        scale = 10000.0

    elif np.isclose(
        scale,
        1.0,
        rtol=1e-7
    ):
        scale = 1.0

    else:
        raise RuntimeError(
            f"ESA score scale e panjohur: {scale}"
        )

    print(
        f"BASELINE VERIFIED 100/100 | "
        f"HV={baseline_hv:.12f} | "
        f"SCALE={scale:g}",
        flush=True
    )

    current_hv = baseline_hv

    seen = {
        tuple(np.round(row["x"], 10))
        for row in best
    }

    # MAP-Elites grid: one elite per niche.
    cells = {}

    accepted = 0
    valid = 0
    invalid = 0
    errors = 0
    attempts = 0

    # ======================================================
    # NICHE QUALITY
    # ======================================================

    def quality(row, preference):

        f1, f2 = row["f"][:2]

        return (
            preference * f1 / REF[0]
            + (1 - preference) * f2 / REF[1]
        )

    def put_cell(row):

        k = niche(row["x"])

        preference = (
            0.25
            + 0.5
            * (
                (k[0] + 2 * k[1] + k[2]) % 7
            ) / 6
        )

        if (
            k not in cells
            or quality(row, preference)
            < quality(cells[k], preference)
        ):

            cells[k] = row
            return True

        return False

    for row in best:
        put_cell(row)

    # ======================================================
    # CHECKPOINT
    # ======================================================

    def checkpoint():

        save_json(
            args.out / "checkpoint.json",
            {
                "hv_unverified": current_hv,
                "baseline_hv": baseline_hv,
                "valid": valid,
                "invalid": invalid,
                "errors": errors,
                "attempts": attempts,
                "accepted": accepted,
                "population": best,
                "archive": list(cells.values())
            }
        )

    # ======================================================
    # ACCEPT CANDIDATES
    # ======================================================

    def consider(row):

        nonlocal best
        nonlocal current_hv
        nonlocal accepted

        if np.any(
            np.asarray(row["f"][:2]) >= REF
        ):
            return

        # Preserve the best configuration
        # in its Walker niche.
        put_cell(row)

        # Check whether it improves submission HV.
        candidate = select100(
            best + [row]
        )

        candidate_hv = hv(candidate)

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
    # IMPORT PREVIOUS ARCHIVE
    # ======================================================

    if args.seed_checkpoint:

        if not args.seed_checkpoint.is_file():
            parser.error(
                "Seed checkpoint nuk ekziston."
            )

        previous = json.loads(
            args.seed_checkpoint.read_text(
                encoding="utf-8"
            )
        )

        imported = (
            previous.get("archive", [])
            + previous.get("population", [])
        )

        print(
            f"Importing {len(imported)} candidates...",
            flush=True
        )

        for item in imported:

            if (
                not isinstance(item, dict)
                or "x" not in item
            ):
                continue

            try:

                x = repair(item["x"])

                k = tuple(np.round(x, 10))

                if k in seen:
                    continue

                if not rover_ok(x):
                    continue

                seen.add(k)

                row = original_row(x)

                consider(row)

            except (
                ValueError,
                TypeError,
                IndexError,
                OverflowError
            ):
                continue

    args.out.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint()

    # ======================================================
    # PARENT SELECTION
    # ======================================================

    def choose_parent():

        # 55% diverse niche elites
        # 45% current best submission.
        pool = (
            list(cells.values())
            if cells and rng.random() < 0.55
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
    # MAP-ELITES MUTATION
    # ======================================================

    def mutate(x):

        z = repair(x)

        kind = int(
            rng.integers(7)
        )

        # 0: Global Walker structure search
        if kind == 0:

            js = rng.choice(
                np.arange(10, 16),
                size=int(rng.integers(1, 5)),
                replace=False
            )

            for j in js:
                z[j] = rng.integers(
                    int(lower[j]),
                    int(upper[j]) + 1
                )

        # 1: Local Walker structure search
        elif kind == 1:

            js = rng.choice(
                np.arange(10, 16),
                size=int(rng.integers(1, 4)),
                replace=False
            )

            for j in js:
                z[j] += rng.choice(
                    [-2, -1, 1, 2]
                )

        # 2: Large orbital geometry change
        elif kind == 2:

            z[:10] += (
                rng.normal(0, 0.12, 10)
                * (upper[:10] - lower[:10])
            )

        # 3: Small orbital geometry change
        elif kind == 3:

            js = rng.choice(
                10,
                size=int(rng.integers(1, 7)),
                replace=False
            )

            z[js] += (
                rng.normal(0, 0.018, len(js))
                * (upper[js] - lower[js])
            )

        # 4: Recombine a Walker constellation
        elif kind == 4:

            other = choose_parent()

            side = int(
                rng.integers(2)
            )

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

        # 5: Rover position search
        elif kind == 5:

            for _ in range(30):

                trial = z.copy()

                js = rng.choice(
                    np.arange(16, 20),
                    size=int(rng.integers(1, 3)),
                    replace=False
                )

                for j in js:
                    trial[j] = rng.integers(
                        int(lower[j]),
                        int(upper[j]) + 1
                    )

                trial = repair(trial)

                if rover_ok(trial):
                    z = trial
                    break

        # 6: Broad hybrid search
        else:

            other = choose_parent()

            z[:10] = (
                other[:10]
                + rng.normal(0, 0.08, 10)
                * (upper[:10] - lower[:10])
            )

            js = rng.choice(
                np.arange(10, 16),
                size=2,
                replace=False
            )

            for j in js:
                z[j] = rng.integers(
                    int(lower[j]),
                    int(upper[j]) + 1
                )

        return repair(z)

    # ======================================================
    # FAST ENGINE
    # ======================================================

    original_graph = udp.build_graph
    original_path = udp.average_shortest_path

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
                "FAST ENGINE: NUMBA + IGRAPH ON",
                flush=True
            )

        # ==================================================
        # QUALITY DIVERSITY OPTIMIZATION
        # ==================================================

        while (
            valid + invalid + errors < args.evaluations
            and attempts < args.evaluations * 15
        ):

            attempts += 1

            x = mutate(
                choose_parent()
            )

            k = tuple(
                np.round(x, 10)
            )

            if k in seen:
                continue

            seen.add(k)

            if not rover_ok(x):
                invalid += 1
                continue

            try:

                f = np.asarray(
                    udp.fitness(x.tolist()),
                    dtype=float
                )

                if (
                    f.shape != (4,)
                    or not np.isfinite(f).all()
                ):
                    errors += 1
                    continue

                if np.any(f[2:] > 0):
                    invalid += 1
                    continue

                valid += 1

                consider({
                    "x": x.tolist(),
                    "f": f.tolist()
                })

            except Exception as exc:

                errors += 1

                if errors <= 3:
                    print(
                        "Fitness error:",
                        repr(exc),
                        flush=True
                    )

            if attempts % 200 == 0:

                checkpoint()

                print(
                    f"Attempts={attempts} | "
                    f"evaluated={valid+invalid+errors}"
                    f"/{args.evaluations} | "
                    f"valid={valid} | "
                    f"invalid={invalid} | "
                    f"cells={len(cells)} | "
                    f"HV={current_hv:.12f}",
                    flush=True
                )

    finally:

        udp.build_graph = original_graph
        udp.average_shortest_path = original_path

    # ======================================================
    # FINAL ORIGINAL ESA VERIFICATION
    # ======================================================

    print(
        "FINAL ESA VERIFICATION 100/100...",
        flush=True
    )

    verified = [
        original_row(row["x"])
        for row in best
    ]

    final_esa_hv = -float(
        combine_scores(
            [row["f"] for row in verified]
        )
    ) / scale

    final_hv = hv(verified)

    if (
        abs(final_esa_hv - final_hv) > 1e-8
        or abs(final_hv - current_hv) > 1e-8
        or final_hv < baseline_hv - 1e-10
    ):
        raise RuntimeError(
            "ESA VERIFICATION FAILED. "
            "Final submission nuk ruhet."
        )

    # ======================================================
    # SAVE VERIFIED SUBMISSION
    # ======================================================

    output_document = copy.deepcopy(document)

    output_payload = (
        output_document[0]
        if isinstance(output_document, list)
        else output_document
    )

    output_payload["decisionVector"] = [
        row["x"]
        for row in verified
    ]

    output = (
        args.out
        / "esa_submission_map_elites_best.json"
    )

    save_json(output, output_document)

    save_json(
        args.out / "verified_result.json",
        {
            "verified_hv": final_hv,
            "baseline_hv": baseline_hv,
            "gain": final_hv - baseline_hv,
            "accepted": accepted,
            "cells": len(cells),
            "valid": valid,
            "invalid": invalid,
            "errors": errors
        }
    )

    checkpoint()

    print(
        f"VERIFIED ESA 100/100 | "
        f"HV={final_hv:.12f} | "
        f"GAIN={final_hv-baseline_hv:.12f} | "
        f"accepted={accepted} | "
        f"cells={len(cells)} | "
        f"errors={errors}",
        flush=True
    )

    print(
        "FILE:",
        output,
        flush=True
    )


if __name__ == "__main__":
    main()
