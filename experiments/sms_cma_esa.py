
"""ESA SpOC 2 - Hybrid CMA-ES with SMS-EMOA selection.

Run from repository root:

python -u -m experiments.sms_cma_esa ^
    --base "C:\ESA_runs\moead_round7\esa_submission_moead_best.json" ^
    --out "C:\ESA_runs\sms_cma_round1" ^
    --evaluations 240 --seed 321

Requires: pip install cma
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

REF = (1.2, 1.4)


def hv(rows):
    points = sorted(
        (float(z["f"][0]), float(z["f"][1]))
        for z in rows
        if z["f"][0] < REF[0] and z["f"][1] < REF[1]
    )
    y_top = REF[1]
    area = 0.0
    for x, y in points:
        if y < y_top:
            area += (REF[0] - x) * (y_top - y)
            y_top = y
    return float(area)


def select_100(rows):
    result = list(rows)
    while len(result) > 100:
        total = hv(result)
        contributions = [
            total - hv(result[:i] + result[i + 1:])
            for i in range(len(result))
        ]
        result.pop(int(np.argmin(contributions)))
    return result


def vector_key(x):
    return tuple(round(float(z), 10) for z in x)


def save_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(content, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    os.replace(tmp, path)


def run(args):
    import cma
    from constellation import udp, combine_scores

    rng = np.random.default_rng(args.seed)
    lb, ub = [
        np.asarray(a, dtype=float)
        for a in udp.get_bounds()
    ]

    if len(lb) != 20 or len(ub) != 20:
        raise ValueError("Modeli duhet te kete 20 parametra.")

    span = ub[:10] - lb[:10]
    if np.any(span <= 0):
        raise ValueError("Kufij te pavlefshem.")

    def normalize(x):
        z = np.clip(np.asarray(x, dtype=float), lb, ub)
        z[10:] = np.rint(z[10:])
        return z.tolist()

    def rover_ok(indices):
        """Fast precheck; full feasibility remains udp.fitness."""
        indices = np.asarray(indices, dtype=int)

        if len(set(indices.tolist())) != 4:
            return False

        if np.any(indices < lb[16:20]):
            return False
        if np.any(indices > ub[16:20]):
            return False

        checker = getattr(udp, "get_rover_constraint", None)
        lambdas = getattr(udp, "lambdas", None)
        phis = getattr(udp, "phis", None)

        if not callable(checker) or lambdas is None or phis is None:
            return True

        try:
            constraint = checker(
                np.asarray(lambdas)[indices],
                np.asarray(phis)[indices]
            )
            c = constraint[0] if isinstance(
                constraint, (tuple, list)
            ) else constraint

            return bool(np.all(np.asarray(c) <= 0))

        except (TypeError, ValueError, IndexError, AttributeError):
            # If the helper API differs, leave the decision to ESA.
            return True

    def evaluate(x):
        x = normalize(x)
        f = np.asarray(udp.fitness(x), dtype=float)

        if f.shape != (4,) or not np.all(np.isfinite(f)):
            return None, 1e6

        penalty = float(np.maximum(f[2:], 0.0).sum())

        if penalty > 0:
            return None, 10.0 + np.log1p(penalty)

        return {"x": x, "f": f.tolist()}, 0.0

    def validate_all(xs):
        rows = []
        for i, x in enumerate(xs):
            row, penalty = evaluate(x)
            if row is None or penalty:
                raise RuntimeError(
                    f"Konfigurimi {i} nuk eshte i vlefshem."
                )
            rows.append(row)
        return rows

    source = json.loads(
        args.base.read_text(encoding="utf-8")
    )
    payload = source[0] if isinstance(source, list) else source
    xs = payload["decisionVector"]

    if len(xs) != 100:
        raise ValueError("Submission duhet te kete 100 vektore.")

    print("Po kontrollohen 100 konfigurimet me ESA...", flush=True)
    best = validate_all(xs)

    def esa_hv(rows):
        return -float(combine_scores(
            [z["f"] for z in rows]
        )) / 10000.0

    initial_hv = esa_hv(best)
    current_hv = hv(best)

    if abs(initial_hv - current_hv) > 1e-8:
        raise RuntimeError(
            "HV lokale nuk perputhet me ESA."
        )

    print(
        f"BASELINE VERIFIED: 100/100 | HV={initial_hv:.12f}",
        flush=True
    )

    args.out.mkdir(parents=True, exist_ok=True)
    output_file = args.out / "esa_submission_sms_cma_best.json"
    checkpoint_file = args.out / "checkpoint.json"

    if output_file.exists():
        previous_doc = json.loads(
            output_file.read_text(encoding="utf-8")
        )
        previous_payload = (
            previous_doc[0]
            if isinstance(previous_doc, list)
            else previous_doc
        )
        previous = validate_all(
            previous_payload["decisionVector"]
        )
        previous_hv = esa_hv(previous)

        if previous_hv > current_hv + 1e-12:
            best = previous
            current_hv = previous_hv
            print(
                f"RESUMED: HV={current_hv:.12f}",
                flush=True
            )

    used = 0
    accepted = 0
    valid = 0
    invalid = 0
    errors = 0
    starts = 0
    seen = {vector_key(z["x"]) for z in best}

    def save():
        document = copy.deepcopy(source)
        target = document[0] if isinstance(document, list) else document
        target["decisionVector"] = [z["x"] for z in best]

        save_json(output_file, document)
        save_json(checkpoint_file, {
            "hv": current_hv,
            "baseline_hv": initial_hv,
            "evaluations": used,
            "accepted": accepted,
            "valid": valid,
            "invalid": invalid,
            "errors": errors,
            "starts": starts,
            "population": best
        })

    save()

    while used + args.popsize <= args.evaluations:
        starts += 1

        template = np.asarray(
            best[int(rng.integers(len(best)))]["x"],
            dtype=float
        ).copy()

        if starts % 4 == 0:
            # Replace 1-2 rovers, but precheck constraints.
            for _ in range(30):
                trial = template.copy()
                indices = rng.choice(
                    np.arange(16, 20),
                    size=int(rng.integers(1, 3)),
                    replace=False
                )

                for j in indices:
                    trial[j] = rng.integers(
                        int(lb[j]), int(ub[j]) + 1
                    )

                if rover_ok(trial[16:20]):
                    template = trial
                    break

        elif starts % 3 == 0:
            # Small discrete S/P/F change.
            j = int(rng.integers(10, 16))
            direction = int(rng.choice([-1, 1]))
            template[j] = np.clip(
                template[j] + direction, lb[j], ub[j]
            )

        normalized = np.clip(
            (template[:10] - lb[:10]) / span,
            1e-6, 1 - 1e-6
        )

        sigma = float(rng.choice([0.025, 0.07, 0.16]))
        weights = rng.dirichlet([1.5, 1.5])

        ideal = np.min(
            np.array([z["f"][:2] for z in best]),
            axis=0
        )

        strategy = cma.CMAEvolutionStrategy(
            normalized.tolist(),
            sigma,
            {
                "bounds": [0, 1],
                "popsize": args.popsize,
                "seed": int(rng.integers(1, 2**30)),
                "verbose": -9,
                "maxiter": args.generations
            }
        )

        for generation in range(args.generations):
            if strategy.stop() or used + args.popsize > args.evaluations:
                break

            candidates = strategy.ask()
            scores = []

            for u in candidates:
                x = template.copy()
                x[:10] = (
                    lb[:10]
                    + np.clip(np.asarray(u), 0, 1) * span
                )
                x = normalize(x)
                used += 1

                k = vector_key(x)
                if k in seen:
                    scores.append(20.0)
                    continue
                seen.add(k)

                if not rover_ok(x[16:20]):
                    invalid += 1
                    scores.append(20.0)
                    continue

                try:
                    row, penalty = evaluate(x)
                except Exception as exc:
                    errors += 1
                    scores.append(20.0)
                    if errors <= 3:
                        print(
                            f"Fitness error: {exc}",
                            flush=True
                        )
                    continue

                if row is None:
                    invalid += 1
                    scores.append(float(penalty))
                    continue

                valid += 1
                f = np.asarray(row["f"][:2])

                scaled = (f - ideal) / np.asarray(REF)
                score = float(
                    np.max(weights * scaled)
                    + 0.02 * np.dot(weights, scaled)
                )
                scores.append(score)

                candidate_population = select_100(
                    best + [row]
                )
                next_hv = hv(candidate_population)

                if next_hv > current_hv + 1e-12:
                    best = candidate_population
                    current_hv = next_hv
                    accepted += 1

                    print(
                        f"IMPROVED #{accepted} | "
                        f"HV={current_hv:.12f} | "
                        f"step={used}",
                        flush=True
                    )
                    save()

            strategy.tell(candidates, scores)

            if used % args.report < args.popsize:
                print(
                    f"Step {used}/{args.evaluations} | "
                    f"valid={valid} | "
                    f"invalid={invalid} | "
                    f"accepted={accepted} | "
                    f"HV={current_hv:.12f}",
                    flush=True
                )
                save()

    print("FINAL ESA VERIFICATION...", flush=True)

    checked = validate_all([z["x"] for z in best])
    verified_hv = esa_hv(checked)

    if (
        abs(verified_hv - current_hv) > 1e-8
        or verified_hv + 1e-9 < initial_hv
    ):
        raise RuntimeError(
            f"Gabim ne verifikim: {current_hv} -> {verified_hv}"
        )

    best = checked
    current_hv = verified_hv
    save()

    print(
        f"VERIFIED ESA 100/100 | HV={current_hv:.12f} | "
        f"GAIN={current_hv-initial_hv:.12f} | "
        f"accepted={accepted} | errors={errors}",
        flush=True
    )
    print("FILE:", output_file, flush=True)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluations", type=int, default=240)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--report", type=int, default=40)

    args = parser.parse_args()

    if (
        args.evaluations < args.popsize
        or args.popsize < 4
        or args.generations < 1
        or args.report < 1
    ):
        parser.error("Parametra te pavlefshem.")

    if not args.base.exists():
        parser.error(f"Mungon fajlli: {args.base}")

    run(args)


if __name__ == "__main__":
    cli()
