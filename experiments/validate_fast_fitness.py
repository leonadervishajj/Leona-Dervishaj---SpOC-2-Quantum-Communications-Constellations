
import json
import time
from pathlib import Path
from types import MethodType

import numpy as np

from constellation import udp
from experiments.test_fast_fitness import fast_average_shortest_path

SOURCE = Path(
    "submission/esa_submission_v4.json"
)


def main():
    document = json.loads(
        SOURCE.read_text(encoding="utf-8")
    )
    payload = (
        document[0]
        if isinstance(document, list)
        else document
    )
    vectors = payload["decisionVector"]

    original_method = udp.average_shortest_path
    original_results = []
    fast_results = []

    try:
        start = time.perf_counter()
        for i, x in enumerate(vectors, 1):
            original_results.append(
                np.asarray(udp.fitness(x), dtype=float)
            )
            if i % 20 == 0:
                print(f"Original {i}/{len(vectors)}", flush=True)
        original_time = time.perf_counter() - start

        udp.average_shortest_path = MethodType(
            fast_average_shortest_path, udp
        )

        start = time.perf_counter()
        for i, x in enumerate(vectors, 1):
            fast_results.append(
                np.asarray(udp.fitness(x), dtype=float)
            )
            if i % 20 == 0:
                print(f"Fast {i}/{len(vectors)}", flush=True)
        fast_time = time.perf_counter() - start

    finally:
        udp.average_shortest_path = original_method

    differences = [
        float(np.max(np.abs(a - b)))
        for a, b in zip(original_results, fast_results)
    ]

    matches = all(
        np.allclose(a, b, rtol=0, atol=1e-8)
        for a, b in zip(original_results, fast_results)
    )

    print("\n--- VERIFIKIMI I PLOTE ---")
    print("Konfigurime:", len(vectors))
    print("Koha origjinale:", round(original_time, 3), "s")
    print("Koha e shpejte:", round(fast_time, 3), "s")
    print("Speedup:", round(original_time / fast_time, 3), "x")
    print("Dallimi maksimal:", max(differences))
    print("PERPUTHET:", matches)

    if not matches:
        raise RuntimeError(
            "Versioni i shpejte nuk perputhet me ESA!"
        )


if __name__ == "__main__":
    main()
