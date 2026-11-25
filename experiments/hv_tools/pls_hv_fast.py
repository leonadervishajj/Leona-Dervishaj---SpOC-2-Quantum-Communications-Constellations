
import argparse
import copy
import json

from collections import deque
from pathlib import Path
from types import MethodType

import numpy as np

from common import ROOT, udp

from experiments.map_elites_esa import (
    REF,
    hv,
    select100
)

from experiments.esa_numba_geometry import fast_build_graph

from experiments.fcmaes_fast_fitness import (
    fast_average_shortest_path
)

def vectors_from_submission(path):

    with path.open(encoding="utf-8") as fh:
        document = json.load(fh)

    payload = (
        document[0]
        if isinstance(document, list)
        else document
    )

    if (
        not isinstance(payload, dict)
        or "decisionVector" not in payload
    ):
        raise ValueError(
            "Expected ESA decisionVector submission format"
        )

    vectors = payload["decisionVector"]

    if len(vectors) != 100:
        raise ValueError(
            "Expected exactly 100 decision vectors"
        )

    return document, vectors

def original_row(x):

    f = np.asarray(
        udp.fitness(
            np.asarray(x, dtype=float).tolist()
        ),
        dtype=float
    )

    if (
        f.shape != (4,)
        or not np.all(np.isfinite(f))
        or np.any(f[2:] > 0)
    ):
        return None

    return {
        "x": np.asarray(x, dtype=float).tolist(),
        "f": f.tolist()
    }

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--base",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--out",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--evaluations",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=191919
    )

    parser.add_argument(
        "--no-fast",
        action="store_true"
    )

    args = parser.parse_args()

    if args.evaluations < 1:
        parser.error(
            "--evaluations must be positive"
        )

    if not args.base.is_file():
        parser.error(
            f"Missing baseline: {args.base}"
        )

    if args.out.exists():
        if (
            not args.out.is_dir()
            or any(args.out.iterdir())
        ):
            parser.error(
                f"Output exists and is not empty: {args.out}"
            )

    rng = np.random.default_rng(args.seed)

    lower, upper = [
        np.asarray(bounds, dtype=float)
        for bounds in udp.get_bounds()
    ]

    if (
        lower.shape != (20,)
        or upper.shape != (20,)
    ):
        raise ValueError(
            "Expected 20-dimensional ESA decision vector"
        )

    width = upper - lower

    document, vectors = vectors_from_submission(
        args.base
    )

    # ==========================================
    # REPAIR
    # ==========================================

    def repair(raw):

        x = np.asarray(raw, dtype=float)

        if (
            x.shape != (20,)
            or not np.isfinite(x).all()
        ):
            raise ValueError(
                "Invalid decision vector"
            )

        z = np.clip(
            x,
            lower,
            upper
        ).copy()

        z[10:] = np.rint(z[10:])

        return z

    # ==========================================
    # ROVER CONSTRAINT
    # ==========================================

    def rover_ok(x):

        ids = np.rint(
            x[16:20]
        ).astype(int)

        if len(set(ids.tolist())) != 4:
            return False

        if (
            np.any(ids < 0)
            or np.any(ids >= len(udp.lambdas))
        ):
            return False

        c, _ = udp.get_rover_constraint(
            udp.lambdas[ids],
            udp.phis[ids]
        )

        return bool(
            np.isfinite(c)
            and c <= 0
        )

    def key(x):
        return tuple(
            np.round(
                np.asarray(x, dtype=float),
                10
            )
        )

    # ==========================================
    # BASELINE VERIFICATION
    # ==========================================

    print(
        "BASELINE: ORIGINAL ESA VERIFICATION 100/100",
        flush=True
    )

    best = []

    for raw in vectors:

        x = repair(raw)

        if not rover_ok(x):
            raise ValueError(
                "Invalid rover selection in baseline"
            )

        row = original_row(x)

        if row is None:
            raise ValueError(
                "Invalid baseline ESA fitness"
            )

        best.append(row)

    best_hv = float(hv(best))

    print(
        f"BASELINE VERIFIED | "
        f"HV={best_hv:.12f} | "
        f"ESA SCORE={-10000 * best_hv:.8f}",
        flush=True
    )

    seen = {
        key(row["x"])
        for row in best
    }

    # ==========================================
    # PLS PARENT SELECTION
    # ==========================================

    def contribution(rows, i, full_hv):

        remaining = (
            rows[:i] + rows[i + 1:]
        )

        return max(
            0.0,
            full_hv - float(hv(remaining))
        )

    parent_queue = deque(
        sorted(
            range(100),
            key=lambda i: contribution(
                best,
                i,
                best_hv
            )
        )[:30]
    )

    successes = 0
    valid = 0
    invalid = 0
    skipped = 0
    fast_disagreements = 0

    old_graph = udp.build_graph
    old_path = udp.average_shortest_path

    # ==========================================
    # FAST ENGINE
    # ==========================================

    def enable_fast():

        udp.build_graph = MethodType(
            fast_build_graph,
            udp
        )

        udp.average_shortest_path = MethodType(
            fast_average_shortest_path,
            udp
        )

    def disable_fast():

        udp.build_graph = old_graph
        udp.average_shortest_path = old_path

    # ==========================================
    # GENERATE NEW CANDIDATE
    # ==========================================

    def propose(parent, iteration):

        z = repair(parent)

        mode = int(
            rng.choice(
                5,
                p=[
                    0.57,
                    0.15,
                    0.12,
                    0.10,
                    0.06
                ]
            )
        )

        # Operator 1: Local orbital search
        # Operator 2: Wider orbital search

        if mode in (0, 1):

            count = int(
                rng.integers(
                    1,
                    5 if mode == 0 else 8
                )
            )

            js = rng.choice(
                np.arange(10),
                count,
                replace=False
            )

            if mode == 0:
                sigmas = [
                    0.001,
                    0.003,
                    0.007,
                    0.015
                ]
            else:
                sigmas = [
                    0.025,
                    0.05
                ]

            sigma = float(
                rng.choice(sigmas)
            )

            z[js] += (
                rng.normal(
                    0,
                    sigma,
                    len(js)
                )
                * width[js]
            )

        # Operator 3: Walker + Orbit

        elif mode == 2:

            side = int(
                rng.integers(2)
            )

            j = int(
                rng.choice(
                    np.arange(
                        10 + 3 * side,
                        13 + 3 * side
                    )
                )
            )

            z[j] += int(
                rng.choice(
                    [-2, -1, 1, 2]
                )
            )

            js = rng.choice(
                np.arange(
                    5 * side,
                    5 * side + 5
                ),
                int(rng.integers(1, 4)),
                replace=False
            )

            z[js] += (
                rng.normal(
                    0,
                    0.01,
                    len(js)
                )
                * width[js]
            )

        # Operator 4: Rover relocation

        elif mode == 3:

            j = int(
                rng.integers(16, 20)
            )

            z[j] = rng.integers(
                int(np.ceil(lower[j])),
                int(np.floor(upper[j])) + 1
            )

        # Operator 5: Walker local move

        else:

            j = int(
                rng.integers(10, 16)
            )

            z[j] += int(
                rng.choice(
                    [-2, -1, 1, 2]
                )
            )

        return repair(z)

    # ==========================================
    # PLS OPTIMIZATION
    # ==========================================

    try:

        if not args.no_fast:

            enable_fast()

            print(
                "FAST ENGINE: ON",
                flush=True
            )

        else:

            print(
                "FAST ENGINE: OFF",
                flush=True
            )

        for step in range(
            1,
            args.evaluations + 1
        ):

            if step % 100 == 0:

                print(
                    f"Progress={step}/{args.evaluations} | "
                    f"valid={valid} | "
                    f"invalid={invalid} | "
                    f"improved={successes} | "
                    f"HV={best_hv:.12f}",
                    flush=True
                )

            # Refresh Pareto priorities

            if (
                not parent_queue
                or step % 250 == 0
            ):

                parent_queue = deque(
                    sorted(
                        range(len(best)),
                        key=lambda i: contribution(
                            best,
                            i,
                            best_hv
                        )
                    )[:30]
                )

            idx = parent_queue[0]

            parent_queue.rotate(-1)

            x = propose(
                best[idx]["x"],
                step
            )

            k = key(x)

            if k in seen:
                skipped += 1
                continue

            seen.add(k)

            if not rover_ok(x):
                invalid += 1
                continue

            try:

                f = np.asarray(
                    udp.fitness(
                        x.tolist()
                    ),
                    dtype=float
                )

            except Exception as exc:

                invalid += 1

                if invalid <= 3:
                    print(
                        "Fitness error:",
                        repr(exc),
                        flush=True
                    )

                continue

            if (
                f.shape != (4,)
                or not np.all(np.isfinite(f))
                or np.any(f[2:] > 0)
            ):
                invalid += 1
                continue

            valid += 1

            if np.any(f[:2] >= REF):
                continue

            fast_row = {
                "x": x.tolist(),
                "f": f.tolist()
            }

            fast_candidate = select100(
                best + [fast_row]
            )

            if (
                float(hv(fast_candidate))
                <= best_hv + 1e-12
            ):
                continue

            # ==================================
            # ORIGINAL ESA VERIFICATION
            # ==================================

            disable_fast()

            try:
                exact_row = original_row(x)

            finally:
                if not args.no_fast:
                    enable_fast()

            if exact_row is None:

                fast_disagreements += 1
                continue

            candidate = select100(
                best + [exact_row]
            )

            new_hv = float(hv(candidate))

            if new_hv <= best_hv + 1e-12:

                fast_disagreements += 1
                continue

            best = candidate
            best_hv = new_hv

            successes += 1

            print(
                f"IMPROVED #{successes} | "
                f"HV={best_hv:.12f} | "
                f"SCORE={-10000 * best_hv:.8f} | "
                f"step={step}",
                flush=True
            )

            new_index = next(
                (
                    i
                    for i, row in enumerate(best)
                    if key(row["x"]) == k
                ),
                idx
            )

            parent_queue.appendleft(
                new_index
            )

    finally:

        disable_fast()

    # ==========================================
    # FINAL ORIGINAL ESA VERIFICATION
    # ==========================================

    print(
        "FINAL ORIGINAL ESA VERIFICATION 100/100",
        flush=True
    )

    verified = []

    for row in best:

        exact = original_row(
            row["x"]
        )

        if exact is None:
            raise RuntimeError(
                "Final ESA verification failed"
            )

        verified.append(exact)

    final_hv = float(
        hv(verified)
    )

    original_baseline = []

    for raw in vectors:

        row = original_row(
            repair(raw)
        )

        if row is None:
            raise RuntimeError(
                "Baseline changed during verification"
            )

        original_baseline.append(row)

    baseline_hv = float(
        hv(original_baseline)
    )

    print(
        f"FINAL | "
        f"baseline={baseline_hv:.12f} | "
        f"new={final_hv:.12f} | "
        f"SCORE={-10000 * final_hv:.8f}",
        flush=True
    )

    print(
        f"Stats: "
        f"attempts={args.evaluations} "
        f"valid={valid} "
        f"invalid={invalid} "
        f"duplicates={skipped} "
        f"fast_mismatch={fast_disagreements}",
        flush=True
    )

    if (
        len({
            key(row["x"])
            for row in verified
        }) != 100
    ):
        raise RuntimeError(
            "Final submission must have "
            "100 distinct vectors"
        )

    if final_hv <= baseline_hv + 1e-12:

        print(
            "NO IMPROVEMENT: "
            "no file written; baseline is untouched",
            flush=True
        )

        return

    # ==========================================
    # SAVE ONLY IMPROVED SUBMISSION
    # ==========================================

    payload = copy.deepcopy(document)

    container = (
        payload[0]
        if isinstance(payload, list)
        else payload
    )

    container["decisionVector"] = [
        row["x"]
        for row in verified
    ]

    args.out.mkdir(
        parents=True,
        exist_ok=True
    )

    result = (
        args.out
        / "pls_improved_submission.json"
    )

    if result.exists():
        raise FileExistsError(
            f"Output already exists: {result}"
        )

    with result.open(
        "x",
        encoding="utf-8"
    ) as fh:

        json.dump(
            payload,
            fh,
            indent=2
        )

    print(
        f"NEW ORIGINAL-VERIFIED RECORD | "
        f"HV={final_hv:.12f} | "
        f"{result}",
        flush=True
    )

if __name__ == "__main__":
    main()