
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from types import MethodType

import numpy as np

from common import udp
from experiments.map_elites_esa import REF, hv, select100
from experiments.esa_numba_geometry import fast_build_graph
from experiments.fcmaes_fast_fitness import fast_average_shortest_path


VERSION = 6

OPERATORS = [
    "orbit_local",
    "orbit_medium",
    "walker_orbit",
    "rover_relocate",
    "walker_local",
    "gap_search",
    "path_relink",
    "structure_jump",
    "gap_bridge",
]

INITIAL_WEIGHTS = np.array(
    [2.5, 1.2, 2.0, 1.0, 0.6, 1.2, 2.0, 2.5, 2.0],
    dtype=float
)

# V6: Lifetime statistics, independent of adaptive resets.
operator_stats = {
    name: {
        "attempts": 0,
        "valid": 0,
        "improvements": 0,
        "total_gain": 0.0,
        "best_gain": 0.0,
    }
    for name in OPERATORS
}


def show_operator_stats():
    print("\n===== V6 OPERATOR ANALYSIS =====")

    for name, s in operator_stats.items():
        attempts = s["attempts"]

        efficiency = (
            s["total_gain"] / attempts
            if attempts > 0 else 0.0
        )

        print(
            f"{name:20s} "
            f"attempts={attempts:6d} "
            f"valid={s['valid']:6d} "
            f"improved={s['improvements']:5d} "
            f"total_gain={s['total_gain']:.10e} "
            f"best_gain={s['best_gain']:.10e} "
            f"gain_per_attempt={efficiency:.10e}"
        )


def load_submission(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    obj = doc[0] if isinstance(doc, list) else doc

    if not isinstance(obj, dict):
        raise ValueError("Invalid ESA JSON structure")

    vectors = obj["decisionVector"]

    if len(vectors) != 100:
        raise ValueError("ESA requires exactly 100 vectors")

    return doc, vectors


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")

    with temp.open("w", encoding="utf-8") as f:
        json.dump(
            value,
            f,
            separators=(",", ":"),
            allow_nan=False
        )
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive PLS V6 with operator statistics for ESA"
    )

    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-fast", action="store_true")

    args = parser.parse_args()

    if args.evaluations <= 0 or args.checkpoint_every <= 0:
        parser.error(
            "Evaluation and checkpoint counts must be positive"
        )

    if not args.base.is_file():
        parser.error(f"Missing baseline: {args.base}")

    checkpoint = args.out / "pls_v6_checkpoint.json"
    result_path = args.out / "pls_improved_submission.json"

    if args.resume:
        if not checkpoint.is_file():
            parser.error(f"No checkpoint: {checkpoint}")
    elif args.out.exists() and any(args.out.iterdir()):
        parser.error(
            "Output folder is not empty. Choose a new --out"
        )

    args.out.mkdir(parents=True, exist_ok=True)

    base_hash = hashlib.sha256(
        args.base.read_bytes()
    ).hexdigest()

    code_hash = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()

    document, input_vectors = load_submission(args.base)

    lower, upper = [
        np.asarray(v, dtype=float)
        for v in udp.get_bounds()
    ]

    if lower.shape != (20,) or upper.shape != (20,):
        raise ValueError("Expected 20-dimensional ESA problem")

    width = upper - lower
    ref = np.asarray(REF, dtype=float)
    rng = np.random.default_rng(args.seed)

    def repair(raw):
        x = np.asarray(raw, dtype=float)

        if x.shape != (20,) or not np.isfinite(x).all():
            raise ValueError("Invalid vector")

        x = np.clip(x, lower, upper).copy()
        x[10:20] = np.rint(x[10:20])

        return x

    def vector_key(raw):
        return tuple(
            np.round(
                np.asarray(raw, dtype=float), 10
            ).tolist()
        )

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
            return bool(np.isfinite(c) and c <= 0)
        except Exception:
            return False

    def evaluate_exact(x):
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
            or not np.isfinite(f).all()
            or np.any(f[2:] > 0)
        ):
            return None

        return {
            "x": np.asarray(x, dtype=float).tolist(),
            "f": f.tolist()
        }

    # BASELINE: Always verify with original ESA.
    print(
        "BASELINE: original ESA verification",
        flush=True
    )

    baseline = []

    for raw in input_vectors:
        x = repair(raw)

        if not rover_ok(x):
            raise ValueError(
                "Invalid rover selection in baseline"
            )

        row = evaluate_exact(x)

        if row is None:
            raise ValueError(
                "Invalid original ESA baseline"
            )

        baseline.append(row)

    if len({
        vector_key(r["x"]) for r in baseline
    }) != 100:
        raise ValueError(
            "Baseline must have 100 distinct vectors"
        )

    baseline_hv = float(hv(baseline))

    print(
        f"BASELINE VERIFIED | HV={baseline_hv:.12f} | "
        f"SCORE={-10000 * baseline_hv:.8f}",
        flush=True
    )

    # FAST ENGINE.
    old_graph = udp.build_graph
    old_path = udp.average_shortest_path

    def fast_on():
        udp.build_graph = MethodType(
            fast_build_graph, udp
        )
        udp.average_shortest_path = MethodType(
            fast_average_shortest_path, udp
        )

    def fast_off():
        udp.build_graph = old_graph
        udp.average_shortest_path = old_path

    # PARETO ANALYSIS.
    def contributions(rows, total_hv):
        result = []

        for i in range(len(rows)):
            reduced = rows[:i] + rows[i + 1:]
            result.append(
                max(
                    0.0,
                    total_hv - float(hv(reduced))
                )
            )

        return np.asarray(result, dtype=float)

    def gap_pairs(rows):
        order = sorted(
            range(len(rows)),
            key=lambda i: rows[i]["f"][1]
        )

        pairs = []

        for left, right in zip(
            order[:-1], order[1:]
        ):
            f1 = rows[left]["f"]
            f2 = rows[right]["f"]

            area = abs(
                (f1[0] - f2[0]) *
                (f1[1] - f2[1])
            )

            pairs.append((area, left, right))

        pairs.sort(reverse=True)

        return pairs[:15]

    def choose_parent(rows, total_hv, gaps):
        style = float(rng.random())

        if style < 0.55:
            c = contributions(rows, total_hv)
            weak = np.argsort(c)[:30]
            return int(rng.choice(weak))

        if style < 0.85 and gaps:
            _, left, right = gaps[
                int(rng.integers(len(gaps)))
            ]

            return int(
                left if rng.random() < 0.5 else right
            )

        return int(rng.integers(len(rows)))

    # ADAPTIVE OPERATOR WEIGHTS.
    weights = INITIAL_WEIGHTS.copy()
    attempts = np.zeros(len(OPERATORS), dtype=int)
    successes = np.zeros(len(OPERATORS), dtype=int)
    rewards = np.zeros(len(OPERATORS), dtype=float)

    def operator_probabilities(stagnation):
        w = np.maximum(weights, 0.15).copy()

        w = 0.85 * w + 0.15 * np.mean(w)

        if stagnation >= 1500:
            w[1] *= 1.6
            w[5] *= 2.0
            w[6] *= 2.0

        if stagnation >= 3500:
            w[1] *= 1.5
            w[5] *= 1.5
            w[6] *= 1.5

        return w / w.sum()

    def adapt_weights():
        nonlocal weights

        success_rate = (
            successes / np.maximum(attempts, 1)
        )

        avg_reward = (
            rewards / np.maximum(attempts, 1)
        )

        gain_signal = np.log1p(
            avg_reward * 1e8
        )

        signal = (
            0.25 * success_rate
            + 0.75 * gain_signal
        )

        new_weights = 0.25 + signal
        new_weights /= np.mean(new_weights)

        weights = (
            0.75 * weights
            + 0.25 * new_weights
        )

        weights = np.clip(weights, 0.20, 6.0)

        attempts.fill(0)
        successes.fill(0)
        rewards.fill(0.0)

    # ALL NINE MUTATION OPERATORS.
    def mutate(parent, op, rows, gaps):
        z = repair(parent)

        if op == "orbit_local":
            n = int(rng.integers(1, 5))
            js = rng.choice(
                10, n, replace=False
            )

            sigma = float(
                rng.choice(
                    [0.001, 0.003, 0.007, 0.015]
                )
            )

            z[js] += (
                rng.normal(0, sigma, n)
                * width[js]
            )

        elif op == "orbit_medium":
            n = int(rng.integers(2, 9))
            js = rng.choice(
                10, n, replace=False
            )

            sigma = float(
                rng.choice(
                    [0.012, 0.025, 0.05]
                )
            )

            z[js] += (
                rng.normal(0, sigma, n)
                * width[js]
            )

        elif op == "walker_orbit":
            side = int(rng.integers(2))

            j = int(
                rng.integers(
                    10 + 3 * side,
                    13 + 3 * side
                )
            )

            z[j] += int(
                rng.choice([-2, -1, 1, 2])
            )

            js = rng.choice(
                np.arange(
                    5 * side, 5 * side + 5
                ),
                size=int(rng.integers(1, 4)),
                replace=False
            )

            z[js] += (
                rng.normal(0, 0.01, len(js))
                * width[js]
            )

        elif op == "rover_relocate":
            j = int(rng.integers(16, 20))

            z[j] = int(
                rng.integers(
                    int(np.ceil(lower[j])),
                    int(np.floor(upper[j])) + 1
                )
            )

        elif op == "walker_local":
            j = int(rng.integers(10, 16))

            z[j] += int(
                rng.choice([-2, -1, 1, 2])
            )

        elif op == "gap_search":
            if gaps:
                _, i, j = gaps[
                    int(rng.integers(len(gaps)))
                ]

                a = np.asarray(
                    rows[i]["x"], dtype=float
                )

                b = np.asarray(
                    rows[j]["x"], dtype=float
                )

                alpha = float(
                    rng.uniform(0.2, 0.8)
                )

                z = (
                    a.copy()
                    if rng.random() < 0.5
                    else b.copy()
                )

                z[:10] = (
                    alpha * a[:10]
                    + (1 - alpha) * b[:10]
                )

                z[:10] += (
                    rng.normal(0, 0.004, 10)
                    * width[:10]
                )

            else:
                z[:10] += (
                    rng.normal(0, 0.01, 10)
                    * width[:10]
                )

        elif op == "path_relink":
            mate = np.asarray(
                rows[
                    int(rng.integers(len(rows)))
                ]["x"],
                dtype=float
            )

            alpha = float(
                rng.uniform(0.10, 0.55)
            )

            js = rng.choice(
                10,
                size=int(rng.integers(2, 9)),
                replace=False
            )

            z[js] = (
                (1 - alpha) * z[js]
                + alpha * mate[js]
            )

            if rng.random() < 0.2:
                j = int(rng.integers(10, 16))

                if mate[j] > z[j]:
                    z[j] += 1
                elif mate[j] < z[j]:
                    z[j] -= 1

        elif op == "structure_jump":
            mate = np.asarray(
                rows[
                    int(rng.integers(len(rows)))
                ]["x"],
                dtype=float
            )

            if rng.random() < 0.5:
                z[10:20] = mate[10:20]

            side = int(rng.integers(2))

            indices = np.arange(
                5 * side,
                5 * side + 5
            )

            alpha = float(
                rng.uniform(0.2, 0.8)
            )

            z[indices] = (
                (1.0 - alpha) * z[indices]
                + alpha * mate[indices]
            )

            count = int(
                rng.integers(2, 6)
            )

            js = rng.choice(
                10,
                size=count,
                replace=False
            )

            sigma = float(
                rng.choice(
                    [0.01, 0.025, 0.05]
                )
            )

            z[js] += (
                rng.normal(0, sigma, count)
                * width[js]
            )

            if rng.random() < 0.35:
                j = int(
                    rng.integers(10, 16)
                )

                z[j] += int(
                    rng.choice([-2, -1, 1, 2])
                )

        elif op == "gap_bridge":
            if gaps:
                areas = np.array(
                    [
                        max(g[0], 0.0)
                        for g in gaps
                    ],
                    dtype=float
                )

                if areas.sum() > 0:
                    probabilities = (
                        areas / areas.sum()
                    )
                else:
                    probabilities = np.full(
                        len(gaps),
                        1.0 / len(gaps)
                    )

                selected = int(
                    rng.choice(
                        len(gaps),
                        p=probabilities
                    )
                )

                _, left, right = gaps[selected]

                a = np.asarray(
                    rows[left]["x"],
                    dtype=float
                )

                b = np.asarray(
                    rows[right]["x"],
                    dtype=float
                )

                z = (
                    a.copy()
                    if rng.random() < 0.5
                    else b.copy()
                )

                alpha = float(
                    rng.uniform(0.15, 0.85)
                )

                z[:10] = (
                    alpha * a[:10]
                    + (1.0 - alpha) * b[:10]
                )

                sigma = float(
                    rng.choice(
                        [0.002, 0.005, 0.015]
                    )
                )

                z[:10] += (
                    rng.normal(0, sigma, 10)
                    * width[:10]
                )

            else:
                z[:10] += (
                    rng.normal(0, 0.01, 10)
                    * width[:10]
                )

        return repair(z)

    # INITIAL STATE.
    best = baseline
    best_hv = baseline_hv

    seen = {
        vector_key(r["x"])
        for r in best
    }

    step_start = 0
    valid = 0
    invalid = 0
    duplicates = 0
    improved = 0
    mismatches = 0
    last_improvement = 0

    # RESUME WITH STATISTICS.
    if args.resume:
        state = json.loads(
            checkpoint.read_text(
                encoding="utf-8"
            )
        )

        if (
            state["version"] != VERSION
            or state["base_hash"] != base_hash
            or state["code_hash"] != code_hash
            or state["seed"] != args.seed
            or state["fast"] != (not args.no_fast)
        ):
            raise ValueError(
                "Checkpoint settings, baseline or code mismatch"
            )

        step_start = int(state["step"])

        if args.evaluations < step_start:
            raise ValueError(
                "Total evaluations cannot be smaller "
                "than the checkpoint step"
            )

        verified = []

        for r in state["best"]:
            x = repair(r["x"])

            if not rover_ok(x):
                raise ValueError(
                    "Invalid checkpoint rover"
                )

            exact = evaluate_exact(x)

            if exact is None:
                raise ValueError(
                    "Invalid checkpoint solution"
                )

            verified.append(exact)

        if (
            len(verified) != 100
            or len({
                vector_key(r["x"])
                for r in verified
            }) != 100
        ):
            raise ValueError(
                "Checkpoint requires 100 distinct solutions"
            )

        best = verified
        best_hv = float(hv(best))

        if abs(
            best_hv - state["best_hv"]
        ) > 1e-10:
            raise ValueError(
                "Checkpoint HV verification failed"
            )

        seen = {
            tuple(k)
            for k in state["seen"]
        }

        weights = np.array(
            state["weights"], dtype=float
        )

        attempts = np.array(
            state["attempts"], dtype=int
        )

        successes = np.array(
            state["successes"], dtype=int
        )

        rewards = np.array(
            state["rewards"], dtype=float
        )

        valid = int(state["valid"])
        invalid = int(state["invalid"])
        duplicates = int(state["duplicates"])
        improved = int(state["improved"])
        mismatches = int(state["mismatches"])
        last_improvement = int(
            state["last_improvement"]
        )

        saved_stats = state.get(
            "operator_stats"
        )

        if (
            saved_stats is None
            or set(saved_stats) != set(OPERATORS)
        ):
            raise ValueError(
                "Missing or incompatible V6 operator statistics"
            )

        for name in OPERATORS:
            s = saved_stats[name]

            operator_stats[name] = {
                "attempts": int(s["attempts"]),
                "valid": int(s["valid"]),
                "improvements": int(
                    s["improvements"]
                ),
                "total_gain": float(
                    s["total_gain"]
                ),
                "best_gain": float(
                    s["best_gain"]
                ),
            }

        rng.bit_generator.state = (
            state["rng_state"]
        )

        print(
            f"RESUMED | "
            f"step={step_start}/{args.evaluations} | "
            f"HV={best_hv:.12f}",
            flush=True
        )

    # CHECKPOINT INCLUDES OPERATOR STATISTICS.
    def save_checkpoint(step):
        state = {
            "version": VERSION,
            "base_hash": base_hash,
            "code_hash": code_hash,
            "seed": args.seed,
            "fast": not args.no_fast,
            "step": step,
            "best": best,
            "best_hv": best_hv,
            "seen": [
                list(k) for k in seen
            ],
            "weights": weights.tolist(),
            "attempts": attempts.tolist(),
            "successes": successes.tolist(),
            "rewards": rewards.tolist(),
            "rng_state": rng.bit_generator.state,
            "valid": valid,
            "invalid": invalid,
            "duplicates": duplicates,
            "improved": improved,
            "mismatches": mismatches,
            "last_improvement": last_improvement,
            "operator_stats": operator_stats,
        }

        atomic_json(checkpoint, state)

        print(
            f"CHECKPOINT SAVED | "
            f"step={step} | HV={best_hv:.12f}",
            flush=True
        )

    # MAIN OPTIMIZATION LOOP.
    last_step = step_start

    try:
        if not args.no_fast:
            fast_on()

        print(
            "FAST ENGINE: "
            + ("OFF" if args.no_fast else "ON"),
            flush=True
        )

        gaps = gap_pairs(best)

        for step in range(
            step_start + 1,
            args.evaluations + 1
        ):
            stagnation = (
                step - last_improvement
            )

            if (
                step == step_start + 1
                or step % 200 == 0
            ):
                gaps = gap_pairs(best)

            if step % 250 == 0:
                adapt_weights()

            probs = operator_probabilities(
                stagnation
            )

            op_id = int(
                rng.choice(
                    len(OPERATORS),
                    p=probs
                )
            )

            op = OPERATORS[op_id]

            attempts[op_id] += 1

            # V6: Count every operator call.
            operator_stats[op]["attempts"] += 1

            parent_id = choose_parent(
                best, best_hv, gaps
            )

            x = mutate(
                best[parent_id]["x"],
                op,
                best,
                gaps
            )

            k = vector_key(x)

            if k in seen:
                duplicates += 1

            else:
                seen.add(k)

                if not rover_ok(x):
                    invalid += 1

                else:
                    try:
                        f = np.asarray(
                            udp.fitness(
                                x.tolist()
                            ),
                            dtype=float
                        )
                    except Exception:
                        f = np.array([])

                    if (
                        f.shape != (4,)
                        or not np.isfinite(f).all()
                        or np.any(f[2:] > 0)
                    ):
                        invalid += 1

                    else:
                        valid += 1

                        # V6: Count valid candidates.
                        operator_stats[op]["valid"] += 1

                        if np.all(
                            f[:2] < ref
                        ):
                            candidate_fast = select100(
                                best + [{
                                    "x": x.tolist(),
                                    "f": f.tolist()
                                }]
                            )

                            fast_hv = float(
                                hv(candidate_fast)
                            )

                            if (
                                fast_hv
                                > best_hv + 1e-12
                            ):
                                # Verify every accepted gain
                                # using the original ESA engine.
                                fast_off()

                                try:
                                    exact = evaluate_exact(x)
                                finally:
                                    if not args.no_fast:
                                        fast_on()

                                if exact is not None:
                                    candidate = select100(
                                        best + [exact]
                                    )

                                    new_hv = float(
                                        hv(candidate)
                                    )

                                    if (
                                        new_hv
                                        > best_hv + 1e-12
                                    ):
                                        gain = (
                                            new_hv - best_hv
                                        )

                                        if (
                                            len(candidate) != 100
                                            or len({
                                                vector_key(r["x"])
                                                for r in candidate
                                            }) != 100
                                        ):
                                            raise RuntimeError(
                                                "Selection returned "
                                                "invalid 100-point set"
                                            )

                                        best = candidate
                                        best_hv = new_hv
                                        improved += 1
                                        last_improvement = step

                                        successes[op_id] += 1
                                        rewards[op_id] += gain

                                        # V6: Real ESA-verified gains.
                                        s = operator_stats[op]

                                        s["improvements"] += 1
                                        s["total_gain"] += float(gain)
                                        s["best_gain"] = max(
                                            s["best_gain"],
                                            float(gain)
                                        )

                                        print(
                                            f"IMPROVED #{improved} | "
                                            f"step={step} | "
                                            f"operator={op} | "
                                            f"gain={gain:.3e} | "
                                            f"HV={best_hv:.12f} | "
                                            f"SCORE={-10000 * best_hv:.8f}",
                                            flush=True
                                        )

                                    else:
                                        mismatches += 1

                                else:
                                    mismatches += 1

            last_step = step

            if step % args.checkpoint_every == 0:
                save_checkpoint(step)

            if step % 500 == 0:
                print(
                    f"Progress={step}/{args.evaluations} | "
                    f"valid={valid} | "
                    f"invalid={invalid} | "
                    f"improved={improved} | "
                    f"stagnation={step-last_improvement} | "
                    f"HV={best_hv:.12f}",
                    flush=True
                )

    except KeyboardInterrupt:
        # Latest periodic checkpoint can be resumed.
        show_operator_stats()

        print(
            f"INTERRUPTED | "
            f"last completed={last_step} | "
            f"resume using --resume",
            flush=True
        )
        return

    finally:
        fast_off()

    # SAVE AND DISPLAY OPERATOR ANALYSIS.
    show_operator_stats()

    atomic_json(
        args.out / "operator_stats.json",
        operator_stats
    )

    # FINAL ESA VERIFICATION.
    print(
        "FINAL ORIGINAL ESA VERIFICATION",
        flush=True
    )

    verified = []

    for r in best:
        x = repair(r["x"])

        if not rover_ok(x):
            raise RuntimeError(
                "Final rover check failed"
            )

        exact = evaluate_exact(x)

        if exact is None:
            raise RuntimeError(
                "Final ESA fitness verification failed"
            )

        verified.append(exact)

    if (
        len(verified) != 100
        or len({
            vector_key(r["x"])
            for r in verified
        }) != 100
    ):
        raise RuntimeError(
            "Final submission must contain "
            "100 distinct valid vectors"
        )

    final_hv = float(hv(verified))

    if abs(
        final_hv - best_hv
    ) > 1e-10:
        raise RuntimeError(
            "Original ESA and optimized HV disagree"
        )

    print(
        "FINAL ORIGINAL ESA VERIFICATION 100/100",
        flush=True
    )

    print(
        f"FINAL | "
        f"baseline={baseline_hv:.12f} | "
        f"new={final_hv:.12f} | "
        f"SCORE={-10000 * final_hv:.8f}",
        flush=True
    )

    save_checkpoint(args.evaluations)

    if final_hv <= baseline_hv + 1e-12:
        print(
            "NO IMPROVEMENT. Original submission preserved.",
            flush=True
        )
        return

    if result_path.exists():
        raise FileExistsError(
            f"Refusing overwrite: {result_path}"
        )

    output = copy.deepcopy(document)
    obj = (
        output[0]
        if isinstance(output, list)
        else output
    )

    obj["decisionVector"] = [
        r["x"] for r in verified
    ]

    with result_path.open(
        "x", encoding="utf-8"
    ) as f:
        json.dump(output, f, indent=2)

    print(
        f"NEW ORIGINAL-VERIFIED RECORD | "
        f"HV={final_hv:.12f} | "
        f"{result_path}",
        flush=True
    )

    print(
        "ADAPTIVE PLS V6 FINISHED",
        flush=True
    )


if __name__ == "__main__":
    main()
