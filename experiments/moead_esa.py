
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission" / "esa_submission_structure.json"
OUT = ROOT / "results" / "moead_esa"
REF = (1.2, 1.4)


def hv2d(rows):
    pts = sorted(
        (float(r["f"][0]), float(r["f"][1]))
        for r in rows
        if np.isfinite(r["f"][:2]).all()
        and r["f"][0] < REF[0]
        and r["f"][1] < REF[1]
    )

    top = REF[1]
    area = 0.0

    for x, y in pts:
        if y < top:
            area += (REF[0] - x) * (top - y)
            top = y

    return float(area)


def select100(rows):
    result = list(rows)

    while len(result) > 100:
        area = hv2d(result)

        contributions = [
            area - hv2d(result[:i] + result[i + 1:])
            for i in range(len(result))
        ]

        result.pop(int(np.argmin(contributions)))

    return result


def normalize_x(x, lb, ub):
    x = np.clip(np.asarray(x, dtype=float), lb, ub)
    x[10:] = np.rint(x[10:])
    return x.tolist()


def xkey(x):
    return tuple(round(float(v), 9) for v in x)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    os.replace(tmp, path)


def scalar(f, weight, ideal):
    vals = np.maximum(
        0,
        (np.asarray(f[:2]) - ideal) / np.asarray(REF)
    )

    return float(
        np.max(weight * vals)
        + 0.01 * np.dot(weight, vals)
    )


def mutate(rng, x, y, z, lb, ub):
    x, y, z = (
        np.asarray(v, dtype=float)
        for v in (x, y, z)
    )

    child = x.copy()
    mode = int(rng.integers(5))

    if mode == 0:
        # Differential Evolution
        mutant = (
            x[:10]
            + rng.uniform(0.3, 0.9) * (y[:10] - z[:10])
        )

        mask = rng.random(10) < 0.65
        mask[int(rng.integers(10))] = True

        child[:10] = np.where(
            mask, mutant, child[:10]
        )

    elif mode == 1:
        # Ndryshime ne gjeometrine Walker
        n = int(rng.integers(2, 7))
        inds = rng.choice(
            10, n, replace=False
        )

        child[inds] += (
            rng.normal(
                0,
                rng.choice([0.015, 0.04, 0.10]),
                n
            )
            * (ub[inds] - lb[inds])
        )

    elif mode == 2:
        # Parametrat diskrete S/P/F
        inds = rng.choice(
            np.arange(10, 16),
            size=int(rng.integers(1, 4)),
            replace=False
        )

        for j in inds:
            if rng.random() < 0.55:
                child[j] += int(
                    rng.choice([-2, -1, 1, 2])
                )
            else:
                child[j] = rng.integers(
                    int(lb[j]),
                    int(ub[j]) + 1
                )

    elif mode == 3:
        # Zevendesimi i rover-eve
        inds = rng.choice(
            np.arange(16, 20),
            size=int(rng.integers(1, 3)),
            replace=False
        )

        for j in inds:
            child[j] = rng.integers(
                int(lb[j]),
                int(ub[j]) + 1
            )

    else:
        # Rikombinimi i nje Walker
        w = int(rng.integers(2))

        child[w * 5:w * 5 + 5] = (
            y[w * 5:w * 5 + 5]
        )

        child[10 + w * 3:13 + w * 3] = (
            y[10 + w * 3:13 + w * 3]
        )

        child[:10] += (
            rng.normal(0, 0.025, 10)
            * (ub[:10] - lb[:10])
        )

    return normalize_x(child, lb, ub), mode


def main(args):
    from constellation import udp, combine_scores

    rng = np.random.default_rng(args.seed)

    lb, ub = [
        np.asarray(t, dtype=float)
        for t in udp.get_bounds()
    ]

    base_data = json.loads(
        args.base.read_text(encoding="utf-8")
    )

    base_payload = (
        base_data[0]
        if isinstance(base_data, list)
        else base_data
    )

    vectors = base_payload["decisionVector"]

    if len(vectors) != 100:
        raise ValueError(
            "Baseline duhet te kete 100 konfigurime."
        )

    args.out.mkdir(parents=True, exist_ok=True)

    output_file = (
        args.out / "esa_submission_moead_best.json"
    )

    checkpoint_file = (
        args.out / "checkpoint.json"
    )

    def validate(xs):
        rows = []

        for idx, x in enumerate(xs):
            f = np.asarray(
                udp.fitness(x),
                dtype=float
            )

            if (
                len(f) != 4
                or not np.all(np.isfinite(f))
                or np.any(f[2:] > 0)
            ):
                raise RuntimeError(
                    f"Vektor i pavlefshem #{idx}: {f}"
                )

            rows.append({
                "x": normalize_x(x, lb, ub),
                "f": f[:2].tolist()
            })

        return rows

    print(
        "Duke verifikuar 100 konfigurimet...",
        flush=True
    )

    best = validate(vectors)

    base_hv = -float(
        combine_scores([
            [*r["f"], 0.0, 0.0]
            for r in best
        ])
    ) / 10000

    current_hv = hv2d(best)

    if abs(base_hv - current_hv) > 1e-7:
        raise RuntimeError(
            "HV nuk perputhet me ESA."
        )

    print(
        f"BASELINE ESA VERIFIED: "
        f"HV={base_hv:.12f}",
        flush=True
    )

    if output_file.exists():
        old = json.loads(
            output_file.read_text(encoding="utf-8")
        )

        old_p = (
            old[0] if isinstance(old, list)
            else old
        )

        old_rows = validate(
            old_p["decisionVector"]
        )

        if hv2d(old_rows) > current_hv + 1e-12:
            best = old_rows
            current_hv = hv2d(best)

            print(
                f"RESUMED: HV={current_hv:.12f}",
                flush=True
            )

    N = int(args.population)

    weights = np.column_stack((
        np.linspace(0.02, 0.98, N),
        np.linspace(0.98, 0.02, N)
    ))

    ideal = np.min(
        np.asarray([r["f"] for r in best]),
        axis=0
    )

    pop = [
        copy.deepcopy(
            min(
                best,
                key=lambda row: scalar(
                    row["f"], w, ideal
                )
            )
        )
        for w in weights
    ]

    distances = np.abs(
        np.arange(N)[:, None]
        - np.arange(N)[None, :]
    )

    neighborhoods = np.argsort(
        distances, axis=1
    )[:, :min(N, args.neighbors)]

    archive = list(best)
    seen = {
        xkey(r["x"]) for r in best
    }

    accepted = 0
    valid = 0
    invalid = 0
    errors = 0

    def save():
        document = copy.deepcopy(base_data)

        payload = (
            document[0]
            if isinstance(document, list)
            else document
        )

        payload["decisionVector"] = [
            r["x"] for r in best
        ]

        save_json(output_file, document)

        save_json(checkpoint_file, {
            "HV": current_hv,
            "accepted": accepted,
            "valid": valid,
            "invalid": invalid,
            "errors": errors,
            "population": best
        })

    save()

    for step in range(args.evaluations):
        i = step % N
        ids = neighborhoods[i]

        if len(ids) >= 3:
            take = rng.choice(
                ids, size=3, replace=False
            )
        else:
            take = rng.choice(
                np.arange(N),
                size=3,
                replace=True
            )

        proposal, mode = mutate(
            rng,
            pop[int(take[0])]["x"],
            pop[int(take[1])]["x"],
            pop[int(take[2])]["x"],
            lb,
            ub
        )

        k = xkey(proposal)

        if k in seen:
            continue

        seen.add(k)

        try:
            f = np.asarray(
                udp.fitness(proposal),
                dtype=float
            )

            if (
                len(f) != 4
                or not np.all(np.isfinite(f))
                or np.any(f[2:] > 0)
            ):
                invalid += 1
                continue

        except Exception as exc:
            errors += 1

            if errors <= 3:
                print(
                    f"Fitness error: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True
                )

            continue

        valid += 1

        row = {
            "x": proposal,
            "f": f[:2].tolist()
        }

        ideal = np.minimum(
            ideal,
            f[:2]
        )

        for j in rng.permutation(ids):
            j = int(j)

            if scalar(
                row["f"],
                weights[j],
                ideal
            ) < scalar(
                pop[j]["f"],
                weights[j],
                ideal
            ):
                pop[j] = row

        if all(
            not (
                np.all(
                    np.asarray(a["f"]) <= f[:2]
                )
                and np.any(
                    np.asarray(a["f"]) < f[:2]
                )
            )
            for a in archive
        ):
            archive = [
                a for a in archive
                if not (
                    np.all(
                        f[:2] <= np.asarray(a["f"])
                    )
                    and np.any(
                        f[:2] < np.asarray(a["f"])
                    )
                )
            ]

            archive.append(row)

        candidate_set = select100(
            best + [row]
        )

        candidate_hv = hv2d(
            candidate_set
        )

        if candidate_hv > current_hv + 1e-12:
            best = candidate_set
            current_hv = candidate_hv
            accepted += 1

            print(
                f"IMPROVED #{accepted}: "
                f"HV={current_hv:.12f} | "
                f"GAIN={current_hv-base_hv:.12f} | "
                f"step={step+1} | mode={mode}",
                flush=True
            )

            save()

        if (step + 1) % args.report == 0:
            print(
                f"Step {step+1}/{args.evaluations} "
                f"| valid={valid} "
                f"| invalid={invalid} "
                f"| accepted={accepted} "
                f"| HV={current_hv:.12f}",
                flush=True
            )

            save()

    print(
        "Verifikimi final ESA...",
        flush=True
    )

    checked = validate([
        r["x"] for r in best
    ])

    checked_hv = -float(
        combine_scores([
            [*r["f"], 0.0, 0.0]
            for r in checked
        ])
    ) / 10000

    if (
        abs(checked_hv - current_hv) > 1e-7
        or checked_hv + 1e-9 < base_hv
    ):
        raise RuntimeError(
            "Mospërputhje ne verifikimin final HV."
        )

    best = checked
    current_hv = checked_hv

    save()

    print(
        f"VERIFIED ESA 100/100 | "
        f"HV={current_hv:.12f} | "
        f"GAIN={current_hv-base_hv:.12f} | "
        f"accepted={accepted} | "
        f"errors={errors}",
        flush=True
    )

    print(
        "FILE:",
        output_file,
        flush=True
    )


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base", type=Path, default=BASE
    )

    parser.add_argument(
        "--out", type=Path, default=OUT
    )

    parser.add_argument(
        "--evaluations", type=int, default=150
    )

    parser.add_argument(
        "--population", type=int, default=36
    )

    parser.add_argument(
        "--neighbors", type=int, default=10
    )

    parser.add_argument(
        "--report", type=int, default=15
    )

    parser.add_argument(
        "--seed", type=int, default=75
    )

    args = parser.parse_args()

    if (
        args.population < 4
        or args.neighbors < 3
        or args.evaluations < 0
    ):
        parser.error(
            "Parametra te pavlefshem."
        )

    if not args.base.exists():
        parser.error(
            f"Mungon baseline: {args.base}"
        )

    main(args)


if __name__ == "__main__":
    cli()
