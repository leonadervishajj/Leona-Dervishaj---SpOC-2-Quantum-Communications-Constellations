
"""Adaptive Large Neighborhood Search for ESA SpOC2 Quantum Communications.

Run from repository root:
python -m experiments.alns_esa --base PATH --out PATH

Existing algorithms and original constellation.py are not modified.
"""

import argparse
import json
from pathlib import Path
from types import MethodType

import numpy as np

from constellation import udp, combine_scores
from experiments.map_elites_esa import REF, hv, select100, save_json
from experiments.esa_numba_geometry import fast_build_graph
from experiments.fcmaes_fast_fitness import fast_average_shortest_path


def load_vectors(path):
    document = json.loads(path.read_text(encoding="utf-8"))

    payload = (
        document[0]
        if isinstance(document, list)
        else document
    )

    vectors = payload["decisionVector"]

    if len(vectors) != 100:
        raise ValueError(
            "Baseline must have exactly 100 decision vectors"
        )

    return vectors


def main(argv=None):

    parser = argparse.ArgumentParser(
        description=__doc__
    )

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

    args = parser.parse_args(argv)

    if not args.base.is_file():
        parser.error(
            "Baseline file does not exist"
        )

    if args.out.exists() and (
        not args.out.is_dir()
        or any(args.out.iterdir())
    ):
        parser.error(
            "Output path exists and is not an empty directory"
        )

    if args.evaluations < 1:
        parser.error(
            "--evaluations must be >= 1"
        )

    rng = np.random.default_rng(args.seed)

    lower, upper = [
        np.asarray(a, dtype=float)
        for a in udp.get_bounds()
    ]

    width = upper - lower

    # ======================================================
    # DECISION VECTOR REPAIR
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
                "Invalid decision vector"
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

        if len(set(ids.tolist())) != 4:
            return False

        c, _ = udp.get_rover_constraint(
            udp.lambdas[ids],
            udp.phis[ids]
        )

        return bool(
            np.isfinite(c)
            and c <= 0
        )

    # ======================================================
    # ESA FITNESS
    # ======================================================

    def evaluate(x):

        f = np.asarray(
            udp.fitness(x.tolist()),
            dtype=float
        )

        if (
            f.shape != (4,)
            or not np.isfinite(f).all()
        ):
            raise ValueError(
                "ESA fitness returned invalid values"
            )

        if np.any(f[2:] > 0):
            return None

        return {
            "x": x.tolist(),
            "f": f.tolist()
        }

    def key(x):
        return tuple(
            np.round(x, 10)
        )

    # ======================================================
    # BASELINE VERIFICATION
    # ======================================================

    print(
        "BASELINE: verifying original ESA 100/100",
        flush=True
    )

    initial = []

    for raw in load_vectors(args.base):

        x = repair(raw)

        if not rover_ok(x):
            raise ValueError(
                "Baseline contains an invalid rover configuration"
            )

        row = evaluate(x)

        if row is None:
            raise ValueError(
                "Baseline contains an infeasible ESA solution"
            )

        initial.append(row)

    best = initial

    baseline_hv = hv(best)

    score = -float(
        combine_scores(
            [r["f"] for r in best]
        )
    )

    scale = (
        score / baseline_hv
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
            f"Unexpected ESA score scaling: {scale}"
        )

    print(
        f"BASELINE VERIFIED | "
        f"HV={baseline_hv:.12f} | "
        f"SCALE={scale:g}",
        flush=True
    )

    current_hv = baseline_hv

    seen = {
        key(r["x"])
        for r in best
    }

    # ======================================================
    # ALNS ADAPTIVE OPERATORS
    # ======================================================

    operators = (
        "walker_reset",
        "walker_local",
        "orbit_local",
        "orbit_large",
        "rover_relocate",
        "constellation_rebuild"
    )

    weights = np.ones(
        len(operators),
        dtype=float
    )

    uses = np.zeros(
        len(operators),
        dtype=int
    )

    successes = np.zeros(
        len(operators),
        dtype=int
    )

    improvements = 0
    evaluated = 0
    valid = 0
    invalid = 0
    errors = 0
    attempts = 0

    archive = list(best)

    # ======================================================
    # DESTROY / REPAIR OPERATORS
    # ======================================================

    def propose(parent, name):

        x = repair(parent)
        z = x.copy()

        # Operator 1: Global Walker reset
        if name == "walker_reset":

            js = rng.choice(
                np.arange(10, 16),
                size=int(rng.integers(1, 5)),
                replace=False
            )

            for j in js:

                z[j] = rng.integers(
                    int(np.ceil(lower[j])),
                    int(np.floor(upper[j])) + 1
                )

        # Operator 2: Local Walker search
        elif name == "walker_local":

            js = rng.choice(
                np.arange(10, 16),
                size=int(rng.integers(1, 4)),
                replace=False
            )

            z[js] += rng.choice(
                [-2, -1, 1, 2],
                size=len(js)
            )

        # Operator 3: Small orbital changes
        elif name == "orbit_local":

            js = rng.choice(
                10,
                size=int(rng.integers(1, 7)),
                replace=False
            )

            z[js] += (
                rng.normal(
                    0,
                    0.018,
                    size=len(js)
                )
                * width[js]
            )

        # Operator 4: Large orbital changes
        elif name == "orbit_large":

            js = rng.choice(
                10,
                size=int(rng.integers(4, 11)),
                replace=False
            )

            z[js] += (
                rng.normal(
                    0,
                    0.12,
                    size=len(js)
                )
                * width[js]
            )

        # Operator 5: Rover relocation
        elif name == "rover_relocate":

            js = rng.choice(
                np.arange(16, 20),
                size=int(rng.integers(1, 4)),
                replace=False
            )

            for j in js:

                z[j] = rng.integers(
                    int(np.ceil(lower[j])),
                    int(np.floor(upper[j])) + 1
                )

        # Operator 6: Rebuild one Walker constellation
        else:

            donor = archive[
                int(rng.integers(len(archive)))
            ]["x"]

            side = int(
                rng.integers(2)
            )

            z[
                5 * side:5 * side + 5
            ] = donor[
                5 * side:5 * side + 5
            ]

            z[
                10 + 3 * side:13 + 3 * side
            ] = donor[
                10 + 3 * side:13 + 3 * side
            ]

            js = np.arange(
                5 * side,
                5 * side + 5
            )

            z[js] += (
                rng.normal(
                    0,
                    0.025,
                    len(js)
                )
                * width[js]
            )

        return repair(z)

    # ======================================================
    # CHECKPOINT
    # ======================================================

    def save_state():

        save_json(
            args.out / "alns_checkpoint.json",
            {
                "algorithm": "adaptive_large_neighborhood_search",
                "seed": args.seed,
                "evaluations_budget": args.evaluations,
                "evaluations_used": evaluated,
                "attempts": attempts,
                "baseline_hv": baseline_hv,
                "current_hv": current_hv,
                "improvements": improvements,
                "valid": valid,
                "invalid": invalid,
                "errors": errors,
                "operators": {
                    name: {
                        "weight": float(weights[i]),
                        "uses": int(uses[i]),
                        "improvements": int(successes[i])
                    }
                    for i, name in enumerate(operators)
                },
                "population": best
            }
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True
    )

    save_state()

    # ======================================================
    # FAST ENGINE
    # ======================================================

    old_graph = udp.build_graph
    old_path = udp.average_shortest_path

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
                "FAST ENGINE: ON",
                flush=True
            )

        # ==================================================
        # ALNS MAIN OPTIMIZATION LOOP
        # ==================================================

        while (
            evaluated < args.evaluations
            and attempts < args.evaluations * 15
        ):

            attempts += 1

            probabilities = (
                weights / weights.sum()
            )

            i = int(
                rng.choice(
                    len(operators),
                    p=probabilities
                )
            )

            uses[i] += 1

            # 80% protected submission
            # 20% diversity archive
            pool = (
                best
                if rng.random() < 0.8
                else archive
            )

            parent = pool[
                int(rng.integers(len(pool)))
            ]["x"]

            x = propose(
                parent,
                operators[i]
            )

            k = key(x)

            if k in seen:
                continue

            seen.add(k)

            if not rover_ok(x):

                invalid += 1

                weights[i] = max(
                    0.2,
                    0.995 * weights[i]
                )

                continue

            evaluated += 1

            reward = 0.0

            try:

                row = evaluate(x)

                if row is None:
                    invalid += 1

                else:

                    valid += 1

                    if np.all(
                        np.asarray(row["f"][:2]) < REF
                    ):

                        # Keep diversity archive
                        if len(archive) < 300:

                            archive.append(row)

                        else:

                            archive[
                                int(
                                    rng.integers(
                                        100,
                                        len(archive)
                                    )
                                )
                            ] = row

                        # Hypervolume optimization
                        candidate = select100(
                            best + [row]
                        )

                        candidate_hv = hv(candidate)

                        if (
                            candidate_hv
                            > current_hv + 1e-12
                        ):

                            best = candidate
                            current_hv = candidate_hv

                            improvements += 1
                            successes[i] += 1

                            reward = 4.0

                            print(
                                f"IMPROVED #{improvements} | "
                                f"HV={current_hv:.12f} | "
                                f"operator={operators[i]}",
                                flush=True
                            )

                            save_state()

                        else:
                            reward = 0.25

            except Exception as exc:

                errors += 1

                if errors <= 3:

                    print(
                        "Fitness error:",
                        repr(exc),
                        flush=True
                    )

            # Adaptive operator weights
            weights[i] = max(
                0.2,
                0.9 * weights[i]
                + 0.1 * (0.2 + reward)
            )

            if evaluated % 200 == 0:

                save_state()

                print(
                    f"Evaluated={evaluated}/{args.evaluations} | "
                    f"valid={valid} | "
                    f"invalid={invalid} | "
                    f"HV={current_hv:.12f}",
                    flush=True
                )

    finally:

        udp.build_graph = old_graph
        udp.average_shortest_path = old_path

    # ======================================================
    # FINAL ORIGINAL ESA VERIFICATION
    # ======================================================

    print(
        "FINAL ORIGINAL ESA VERIFICATION 100/100",
        flush=True
    )

    verified = []

    for row in best:

        x = repair(row["x"])

        result = evaluate(x)

        if result is None:
            raise RuntimeError(
                "Final submission failed original ESA verification"
            )

        verified.append(result)

    final_hv = hv(verified)

    if not np.isclose(
        final_hv,
        current_hv,
        atol=1e-10,
        rtol=0
    ):
        raise RuntimeError(
            f"Fast/original evaluation mismatch: "
            f"{current_hv} vs {final_hv}"
        )

    current_hv = final_hv

    save_state()

    # ======================================================
    # SAVE IMPROVED SUBMISSION ONLY
    # ======================================================

    if final_hv > baseline_hv + 1e-12:

        original_document = json.loads(
            args.base.read_text(
                encoding="utf-8"
            )
        )

        payload = (
            original_document[0]
            if isinstance(original_document, list)
            else original_document
        )

        payload["decisionVector"] = [
            row["x"]
            for row in verified
        ]

        save_json(
            args.out / "alns_improved_submission.json",
            original_document
        )

        print(
            f"NEW VERIFIED RECORD | "
            f"HV={final_hv:.12f}",
            flush=True
        )

    else:

        print(
            f"NO IMPROVEMENT | "
            f"baseline preserved | "
            f"HV={final_hv:.12f}",
            flush=True
        )


if __name__ == "__main__":
    main()
