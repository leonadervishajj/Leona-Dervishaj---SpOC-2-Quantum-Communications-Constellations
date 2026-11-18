

import argparse
import json
import time
from pathlib import Path
from types import MethodType

import numpy as np

from constellation import udp

from experiments.esa_numba_geometry import (
    fast_build_graph
)

from experiments.fcmaes_fast_fitness import (
    fast_average_shortest_path
)


def evaluate_batch(vectors):
    results = []

    start = time.perf_counter()

    for i, x in enumerate(vectors, 1):
        f = np.asarray(
            udp.fitness(x),
            dtype=float
        )

        results.append(f)

        print(
            f"Evaluated {i}/{len(vectors)}",
            flush=True
        )

    elapsed = time.perf_counter() - start

    return results, elapsed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=20
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/combined_fast_test"
        )
    )

    args = parser.parse_args()

    if args.samples < 1:
        parser.error(
            "samples must be >= 1"
        )

    data = json.loads(
        args.base.read_text(
            encoding="utf-8"
        )
    )

    submission = (
        data[0]
        if isinstance(data, list)
        else data
    )

    vectors = submission[
        "decisionVector"
    ][:args.samples]

    if not vectors:
        raise RuntimeError(
            "No decision vectors found."
        )

    original_graph = udp.build_graph
    original_paths = udp.average_shortest_path

    try:
        print(
            "\nSTEP 1: ORIGINAL ESA",
            flush=True
        )

        original_results, original_time = (
            evaluate_batch(vectors)
        )

        print(
            "\nSTEP 2: NUMBA + IGRAPH",
            flush=True
        )

        udp.build_graph = MethodType(
            fast_build_graph,
            udp
        )

        udp.average_shortest_path = MethodType(
            fast_average_shortest_path,
            udp
        )

        # Warm up Numba before measuring.
        warmup = np.asarray(
            udp.fitness(vectors[0]),
            dtype=float
        )

        if not np.allclose(
            warmup,
            original_results[0],
            rtol=1e-9,
            atol=1e-9
        ):
            raise RuntimeError(
                "Warmup mismatch with ESA!"
            )

        fast_results, fast_time = (
            evaluate_batch(vectors)
        )

    finally:
        # Always restore original ESA methods.
        udp.build_graph = original_graph
        udp.average_shortest_path = original_paths

    differences = []

    matches = True

    for original, fast in zip(
        original_results,
        fast_results
    ):
        if (
            original.shape != (4,)
            or fast.shape != (4,)
            or not np.isfinite(original).all()
            or not np.isfinite(fast).all()
        ):
            matches = False
            differences.append(float("inf"))
            continue

        diff = float(
            np.max(
                np.abs(original - fast)
            )
        )

        differences.append(diff)

        if not np.allclose(
            original,
            fast,
            rtol=1e-9,
            atol=1e-9
        ):
            matches = False

    speedup = original_time / fast_time

    report = {
        "samples": len(vectors),
        "original_seconds": original_time,
        "combined_seconds": fast_time,
        "original_evals_per_second": (
            len(vectors) / original_time
        ),
        "combined_evals_per_second": (
            len(vectors) / fast_time
        ),
        "speedup": speedup,
        "max_abs_difference": max(differences),
        "results_match": matches
    }

    args.out.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        args.out / "benchmark.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)

    print("COMBINED BENCHMARK RESULTS")

    print("=" * 60)

    print(
        json.dumps(report, indent=2),
        flush=True
    )

    if not matches:
        raise RuntimeError(
            "FAILED: Combined results do not "
            "match original ESA fitness."
        )

    print(
        "\nPASS: All objectives and "
        "constraints match ESA.",
        flush=True
    )


if __name__ == "__main__":
    main()
