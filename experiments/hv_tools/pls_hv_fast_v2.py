
import argparse
import copy
import hashlib
import json
import os
from collections import deque
from pathlib import Path
from types import MethodType

import numpy as np

from common import udp
from experiments.map_elites_esa import REF, hv, select100
from experiments.esa_numba_geometry import fast_build_graph
from experiments.fcmaes_fast_fitness import fast_average_shortest_path


def load_document(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    obj = document[0] if isinstance(document, list) else document

    if not isinstance(obj, dict) or "decisionVector" not in obj:
        raise ValueError("Submission needs decisionVector")

    vectors = obj["decisionVector"]

    if len(vectors) != 100:
        raise ValueError("ESA submission must have exactly 100 vectors")

    return document, vectors


def atomic_json(path, data):
    """Old checkpoint survives interrupted writing."""

    temp = path.with_name(path.name + ".tmp")

    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), allow_nan=False)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp, path)


def main():

    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)

    p.add_argument(
        "--evaluations",
        type=int,
        required=True,
        help="Total number of attempts, not additional attempts"
    )

    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-fast", action="store_true")

    a = p.parse_args()

    if a.evaluations < 1 or a.checkpoint_every < 1:
        p.error("--evaluations and --checkpoint-every must be positive")

    if not a.base.is_file():
        p.error(f"Missing input: {a.base}")

    chk = a.out / "pls_checkpoint.json"
    result_file = a.out / "pls_improved_submission.json"

    if a.resume:
        if not chk.is_file():
            p.error(f"No checkpoint available: {chk}")

    elif a.out.exists() and any(a.out.iterdir()):
        p.error(
            f"Output folder not empty: {a.out}. "
            "Choose another --out"
        )

    a.out.mkdir(parents=True, exist_ok=True)

    document, vectors = load_document(a.base)

    digest = hashlib.sha256(a.base.read_bytes()).hexdigest()

    lower, upper = (
        np.asarray(v, dtype=float)
        for v in udp.get_bounds()
    )

    if lower.shape != (20,) or upper.shape != (20,):
        raise ValueError("Expected ESA 20-dimensional vectors")

    width = upper - lower

    ref = np.asarray(REF, dtype=float)

    rng = np.random.default_rng(a.seed)

    # ==========================================
    # REPAIR
    # ==========================================

    def repair(raw):

        x = np.asarray(raw, dtype=float)

        if x.shape != (20,) or not np.isfinite(x).all():
            raise ValueError("Bad decision vector")

        x = np.clip(x, lower, upper).copy()
        x[10:] = np.rint(x[10:])

        return x

    def key(x):
        return tuple(
            np.round(
                np.asarray(x, dtype=float),
                10
            ).tolist()
        )

    # ==========================================
    # ROVER CONSTRAINTS
    # ==========================================

    def rover_ok(x):

        ids = np.rint(x[16:20]).astype(int)

        if len(set(ids.tolist())) != 4:
            return False

        if np.any(ids < 0) or np.any(ids >= len(udp.lambdas)):
            return False

        try:

            c, _ = udp.get_rover_constraint(
                udp.lambdas[ids],
                udp.phis[ids]
            )

            return bool(
                np.isfinite(c)
                and c <= 0
            )

        except Exception:
            return False

    # ==========================================
    # ORIGINAL ESA FITNESS
    # ==========================================

    def exact_row(x):

        try:

            f = np.asarray(
                udp.fitness(
                    np.asarray(x, dtype=float).tolist()
                ),
                dtype=float
            )

        except Exception:
            return None

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

    # ==========================================
    # BASELINE VERIFICATION
    # ==========================================

    print(
        "BASELINE: verifying original ESA 100/100",
        flush=True
    )

    baseline = []

    for raw in vectors:

        x = repair(raw)

        if not rover_ok(x):
            raise ValueError(
                "Invalid baseline rover selection"
            )

        r = exact_row(x)

        if r is None:
            raise ValueError(
                "Invalid ESA baseline"
            )

        baseline.append(r)

    baseline_hv = float(hv(baseline))

    print(
        f"BASELINE VERIFIED | "
        f"HV={baseline_hv:.12f} | "
        f"SCORE={-10000 * baseline_hv:.8f}",
        flush=True
    )

    # ==========================================
    # HV-GUIDED PARENT SELECTION
    # ==========================================

    def contrib(rows, i, total_hv):

        remaining = rows[:i] + rows[i + 1:]

        return max(
            0.0,
            total_hv - float(hv(remaining))
        )

    def initial_queue(rows, total_hv):

        ordered = sorted(
            range(len(rows)),
            key=lambda i: contrib(
                rows,
                i,
                total_hv
            )
        )

        return deque(ordered[:30])

    # ==========================================
    # CANDIDATE GENERATION
    # ==========================================

    def propose(raw):

        z = repair(raw)

        mode = int(
            rng.choice(
                5,
                p=[0.57, 0.15, 0.12, 0.10, 0.06]
            )
        )

        # Local orbital / wider orbital search
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

            sigmas = (
                [0.001, 0.003, 0.007, 0.015]
                if mode == 0
                else [0.025, 0.05]
            )

            sigma = float(rng.choice(sigmas))

            z[js] += (
                rng.normal(0, sigma, len(js))
                * width[js]
            )

        # Walker + Orbit
        elif mode == 2:

            side = int(rng.integers(2))

            j = int(
                rng.choice(
                    np.arange(
                        10 + 3 * side,
                        13 + 3 * side
                    )
                )
            )

            z[j] += int(
                rng.choice([-2, -1, 1, 2])
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
                rng.normal(0, 0.01, len(js))
                * width[js]
            )

        # Rover relocation
        elif mode == 3:

            j = int(rng.integers(16, 20))

            z[j] = rng.integers(
                int(np.ceil(lower[j])),
                int(np.floor(upper[j])) + 1
            )

        # Walker local move
        else:

            j = int(rng.integers(10, 16))

            z[j] += int(
                rng.choice([-2, -1, 1, 2])
            )

        return repair(z)

    # ==========================================
    # INITIAL STATE
    # ==========================================

    best = baseline
    best_hv = baseline_hv

    seen = {
        key(r["x"])
        for r in best
    }

    q = initial_queue(best, best_hv)

    start_step = 0

    valid = 0
    invalid = 0
    duplicates = 0
    improvements = 0
    mismatches = 0

    # ==========================================
    # RESUME FROM CHECKPOINT
    # ==========================================

    if a.resume:

        state = json.loads(
            chk.read_text(encoding="utf-8")
        )

        if (
            state.get("version") != 2
            or state.get("base_sha256") != digest
        ):
            raise ValueError(
                "Checkpoint belongs to a different "
                "code version or baseline JSON"
            )

        if (
            state.get("seed") != a.seed
            or state.get("fast") != (not a.no_fast)
        ):
            raise ValueError(
                "Resume requires the same seed "
                "and fast-engine setting"
            )

        start_step = int(state["step"])

        if a.evaluations < start_step:
            p.error(
                f"--evaluations {a.evaluations} "
                f"< checkpoint step {start_step}"
            )

        best = state["best"]

        if (
            len(best) != 100
            or len({
                key(r["x"])
                for r in best
            }) != 100
        ):
            raise ValueError(
                "Checkpoint does not contain "
                "100 distinct solutions"
            )

        # Verify checkpoint with original ESA fitness
        verified = []

        for r in best:

            exact = exact_row(r["x"])

            if (
                exact is None
                or not rover_ok(
                    np.asarray(r["x"], dtype=float)
                )
            ):
                raise ValueError(
                    "Checkpoint verification failed"
                )

            verified.append(exact)

        actual_hv = float(hv(verified))

        if (
            abs(
                actual_hv
                - float(state["best_hv"])
            ) > 1e-10
        ):
            raise ValueError(
                "Checkpoint HV differs from "
                "ORIGINAL ESA HV"
            )

        best = verified
        best_hv = actual_hv

        seen = {
            tuple(k)
            for k in state["seen"]
        }

        q = deque(state["queue"])

        rng.bit_generator.state = state["rng_state"]

        (
            valid,
            invalid,
            duplicates,
            improvements,
            mismatches
        ) = (
            int(state[name])
            for name in (
                "valid",
                "invalid",
                "duplicates",
                "improvements",
                "mismatches"
            )
        )

        print(
            f"RESUMED | "
            f"step={start_step}/{a.evaluations} | "
            f"HV={best_hv:.12f}",
            flush=True
        )

    # ==========================================
    # SAVE CHECKPOINT
    # ==========================================

    def save_checkpoint(step):

        state = {
            "version": 2,
            "base_sha256": digest,
            "seed": a.seed,
            "fast": not a.no_fast,
            "step": step,
            "best": best,
            "best_hv": best_hv,
            "seen": [list(k) for k in seen],
            "queue": list(q),
            "rng_state": rng.bit_generator.state,
            "valid": valid,
            "invalid": invalid,
            "duplicates": duplicates,
            "improvements": improvements,
            "mismatches": mismatches
        }

        atomic_json(chk, state)

        print(
            f"CHECKPOINT SAVED | "
            f"step={step} | "
            f"HV={best_hv:.12f}",
            flush=True
        )

    # ==========================================
    # FAST ENGINE
    # ==========================================

    old_graph = udp.build_graph
    old_path = udp.average_shortest_path

    def fast_on():

        udp.build_graph = MethodType(
            fast_build_graph,
            udp
        )

        udp.average_shortest_path = MethodType(
            fast_average_shortest_path,
            udp
        )

    def fast_off():

        udp.build_graph = old_graph
        udp.average_shortest_path = old_path

    # ==========================================
    # OPTIMIZATION LOOP
    # ==========================================

    last_complete_step = start_step

    try:

        if not a.no_fast:
            fast_on()

        print(
            "FAST ENGINE: "
            + ("OFF" if a.no_fast else "ON"),
            flush=True
        )

        for step in range(
            start_step + 1,
            a.evaluations + 1
        ):

            if not q or step % 250 == 0:
                q = initial_queue(best, best_hv)

            idx = q[0]
            q.rotate(-1)

            x = propose(
                best[idx]["x"]
            )

            k = key(x)

            if k in seen:
                duplicates += 1

            else:

                seen.add(k)

                if not rover_ok(x):
                    invalid += 1

                else:

                    try:
                        f = np.asarray(
                            udp.fitness(x.tolist()),
                            dtype=float
                        )

                    except Exception:
                        f = np.array([])

                    if (
                        f.shape != (4,)
                        or not np.all(np.isfinite(f))
                        or np.any(f[2:] > 0)
                    ):
                        invalid += 1

                    else:

                        valid += 1

                        if np.all(f[:2] < ref):

                            candidate = select100(
                                best + [{
                                    "x": x.tolist(),
                                    "f": f.tolist()
                                }]
                            )

                            if (
                                float(hv(candidate))
                                > best_hv + 1e-12
                            ):

                                # ORIGINAL ESA CHECK
                                fast_off()

                                try:
                                    exact = exact_row(x)

                                finally:
                                    if not a.no_fast:
                                        fast_on()

                                if exact is not None:

                                    candidate = select100(
                                        best + [exact]
                                    )

                                    new_hv = float(
                                        hv(candidate)
                                    )

                                    if new_hv > best_hv + 1e-12:

                                        best = candidate
                                        best_hv = new_hv

                                        improvements += 1

                                        new_idx = next(
                                            (
                                                i
                                                for i, r in enumerate(best)
                                                if key(r["x"]) == k
                                            ),
                                            idx
                                        )

                                        q.appendleft(new_idx)

                                        print(
                                            f"IMPROVED #{improvements} | "
                                            f"step={step} | "
                                            f"HV={best_hv:.12f} | "
                                            f"SCORE={-10000 * best_hv:.8f}",
                                            flush=True
                                        )

                                    else:
                                        mismatches += 1

                                else:
                                    mismatches += 1

            last_complete_step = step

            if step % a.checkpoint_every == 0:
                save_checkpoint(step)

            if step % 100 == 0:

                print(
                    f"Progress={step}/{a.evaluations} | "
                    f"valid={valid} | "
                    f"invalid={invalid} | "
                    f"improvements={improvements} | "
                    f"HV={best_hv:.12f}",
                    flush=True
                )

    except KeyboardInterrupt:

        print(
            f"INTERRUPTED | "
            f"last complete step={last_complete_step}; "
            f"resume from last CHECKPOINT in {chk}",
            flush=True
        )

        return

    finally:

        fast_off()

    # ==========================================
    # FINAL ORIGINAL ESA VERIFICATION
    # ==========================================

    print(
        "FINAL ORIGINAL ESA VERIFICATION 100/100",
        flush=True
    )

    verified = []

    for r in best:

        exact = exact_row(r["x"])

        if exact is None:
            raise RuntimeError(
                "Final original ESA verification failed"
            )

        verified.append(exact)

    final_hv = float(hv(verified))

    if len({
        key(r["x"])
        for r in verified
    }) != 100:
        raise RuntimeError(
            "Submission must have exactly "
            "100 distinct vectors"
        )

    if abs(final_hv - best_hv) > 1e-10:
        raise RuntimeError(
            "FAST/ORIGINAL ESA metric disagreement: "
            "do not promote"
        )

    print(
        f"FINAL | "
        f"baseline={baseline_hv:.12f} | "
        f"new={final_hv:.12f} | "
        f"SCORE={-10000 * final_hv:.8f}",
        flush=True
    )

    save_checkpoint(a.evaluations)

    if final_hv <= baseline_hv + 1e-12:

        print(
            "NO IMPROVEMENT: original submission preserved",
            flush=True
        )

        return

    if result_file.exists():
        raise FileExistsError(
            f"Result exists, refusing overwrite: {result_file}"
        )

    # ==========================================
    # SAVE VERIFIED SUBMISSION
    # ==========================================

    payload = copy.deepcopy(document)

    obj = (
        payload[0]
        if isinstance(payload, list)
        else payload
    )

    obj["decisionVector"] = [
        r["x"]
        for r in verified
    ]

    with result_file.open(
        "x",
        encoding="utf-8"
    ) as f:

        json.dump(
            payload,
            f,
            indent=2
        )

    print(
        f"NEW ORIGINAL-VERIFIED RECORD | "
        f"HV={final_hv:.12f} | "
        f"{result_file}",
        flush=True
    )


if __name__ == "__main__":
    main()
