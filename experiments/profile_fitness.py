
import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path
from types import MethodType

import numpy as np

from constellation import udp


def run_batch(vectors, profile=False):
    profiler = cProfile.Profile()

    if profile:
        profiler.enable()

    start = time.perf_counter()

    rows = [
        np.asarray(udp.fitness(x), dtype=float)
        for x in vectors
    ]

    duration = time.perf_counter() - start

    if profile:
        profiler.disable()

    return rows, duration, profiler


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base", type=Path, required=True
    )
    parser.add_argument(
        "--samples", type=int, default=8
    )
    parser.add_argument(
        "--top", type=int, default=20
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/fitness_profile")
    )

    args = parser.parse_args()

    if args.samples < 1:
        parser.error("Samples duhet te jete >= 1")

    raw = json.loads(
        args.base.read_text(encoding="utf-8")
    )

    submission = (
        raw[0] if isinstance(raw, list) else raw
    )

    vectors = submission["decisionVector"]
    vectors = vectors[:args.samples]

    args.out.mkdir(parents=True, exist_ok=True)

    print(
        f"Testing {len(vectors)} ESA vectors",
        flush=True
    )

    original = udp.average_shortest_path

    try:
        base_rows, base_time, profiler = run_batch(
            vectors, profile=True
        )

        stream = io.StringIO()

        pstats.Stats(
            profiler, stream=stream
        ).strip_dirs().sort_stats(
            "cumtime"
        ).print_stats(args.top)

        report = stream.getvalue()

        (args.out / "original_profile.txt").write_text(
            report, encoding="utf-8"
        )

        print(report, flush=True)

        from experiments.fcmaes_fast_fitness import (
            fast_average_shortest_path
        )

        udp.average_shortest_path = MethodType(
            fast_average_shortest_path, udp
        )

        fast_rows, fast_time, _ = run_batch(vectors)

    finally:
        udp.average_shortest_path = original

    errors = [
        float(np.max(np.abs(a - b)))
        for a, b in zip(base_rows, fast_rows)
    ]

    matching = all(
        np.allclose(
            a, b, rtol=1e-9, atol=1e-9
        )
        for a, b in zip(base_rows, fast_rows)
    )

    result = {
        "samples": len(vectors),
        "original_seconds": base_time,
        "igraph_seconds": fast_time,
        "original_evals_per_second": (
            len(vectors) / base_time
        ),
        "igraph_evals_per_second": (
            len(vectors) / fast_time
        ),
        "speedup": base_time / fast_time,
        "max_abs_difference": max(errors),
        "results_match": matching
    }

    (args.out / "benchmark.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    print(
        "\nBENCHMARK RESULTS:",
        json.dumps(result, indent=2),
        flush=True
    )

    if not matching:
        raise RuntimeError(
            "STOP: Results differ from ESA!"
        )


if __name__ == "__main__":
    main()
