
import json
import time
from types import MethodType
from pathlib import Path

import numpy as np
import networkx as nx

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from constellation import udp


SOURCE = Path(
    "results/pareto_hv_v3/esa_submission_best.json"
)


# Version me i shpejte i llogaritjes se
# rrugeve mothership -> rover.
#
# Perdoret i njejti graf, te njejtat pesha
# dhe i njejti penalitet 10000 per rruget
# qe nuk ekzistojne.

def fast_average_shortest_path(
    self, G, src, dst, epoch, verbose=False
):
    n = G.number_of_nodes()

    mothership_start = n - src - dst
    rover_start = n - dst

    sources = np.arange(
        mothership_start,
        mothership_start + src
    )

    # Kthejme graf-in NetworkX ne matrice sparse.
    adjacency = nx.to_numpy_array(
        G,
        weight="weight",
        dtype=np.float64
    )

    sparse_graph = csr_matrix(adjacency)

    # Nje ekzekutim i Dijkstra-s per
    # te gjitha 7 motherships.
    distances = dijkstra(
        csgraph=sparse_graph,
        directed=False,
        indices=sources
    )

    # Marrim distancat deri te 4 roveret.
    rover_distances = distances[
        :,
        rover_start:rover_start + dst
    ].copy()

    # I njejti penalitet si ne kodin ESA.
    rover_distances[
        ~np.isfinite(rover_distances)
    ] = 10000.0

    return float(
        np.sum(rover_distances) / src / dst
    )


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Mungon submission-i: {SOURCE}"
        )

    document = json.loads(
        SOURCE.read_text(encoding="utf-8")
    )

    payload = (
        document[0]
        if isinstance(document, list)
        else document
    )

    vectors = payload["decisionVector"][:5]

    print("=" * 60)
    print("ESA - TEST I PERESHPEJTIMIT TE FITNESS")
    print("=" * 60)

    original_method = udp.average_shortest_path

    original_fitness = []
    fast_fitness = []

    try:
        # TESTI 1: MODELI ORIGJINAL
        print("\n1. Duke testuar modelin origjinal...")

        start = time.perf_counter()

        for i, x in enumerate(vectors, 1):
            result = np.asarray(
                udp.fitness(x), dtype=float
            )
            original_fitness.append(result)
            print(f"Origjinal {i}/5", flush=True)

        original_time = time.perf_counter() - start

        # TESTI 2: MODELI I PERESHPEJTUAR
        udp.average_shortest_path = MethodType(
            fast_average_shortest_path,
            udp
        )

        print("\n2. Duke testuar versionin e shpejte...")

        start = time.perf_counter()

        for i, x in enumerate(vectors, 1):
            result = np.asarray(
                udp.fitness(x), dtype=float
            )
            fast_fitness.append(result)
            print(f"Fast {i}/5", flush=True)

        fast_time = time.perf_counter() - start

    finally:
        # Rikthejme funksionin origjinal.
        udp.average_shortest_path = original_method

    # KONTROLLI I SAKTESISE
    differences = [
        np.max(np.abs(a - b))
        for a, b in zip(original_fitness, fast_fitness)
    ]

    max_difference = float(max(differences))

    correct = all(
        np.allclose(a, b, atol=1e-8, rtol=0)
        for a, b in zip(original_fitness, fast_fitness)
    )

    print("\n" + "=" * 60)
    print("REZULTATI I TESTIT")
    print("=" * 60)

    print(f"Koha origjinale: {original_time:.3f} s")
    print(f"Koha e shpejte:  {fast_time:.3f} s")

    if fast_time > 0:
        print(
            f"Speedup: {original_time / fast_time:.2f}x"
        )

    print(
        f"Dallimi maksimal: {max_difference:.12g}"
    )

    print(f"Rezultatet perputhen: {correct}")

    if not correct:
        print(
            "NUK PRANOHET: rezultatet ndryshojne!"
        )
    elif fast_time < original_time:
        print(
            "SUKSES: i njejti fitness me pak kohe."
        )
    else:
        print(
            "Rezultatet jane te sakta, "
            "por ky version nuk eshte me i shpejte."
        )


if __name__ == "__main__":
    main()
