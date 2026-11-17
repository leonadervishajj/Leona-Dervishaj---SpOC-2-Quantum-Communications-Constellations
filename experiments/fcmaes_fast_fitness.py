
import argparse
import json
import time
from pathlib import Path
from types import MethodType

import igraph as ig
import numpy as np

from constellation import udp

def fast_average_shortest_path(
    self, G, src, dst, *args, **kwargs
):
    """Llogarit rruget me igraph ne vend te NetworkX."""

    nodes = list(G.nodes())
    n = len(nodes)

    node_index = {
        node: i for i, node in enumerate(nodes)
    }

    graph = ig.Graph(
        n=n,
        directed=G.is_directed()
    )

    edges = []
    weights = []

    for u, v, attributes in G.edges(data=True):
        edges.append((
            node_index[u],
            node_index[v]
        ))

        weights.append(
            float(attributes.get("weight", 1.0))
        )

    if edges:
        graph.add_edges(edges)

    sources = list(range(
        n - src - dst,
        n - dst
    ))

    targets = list(range(
        n - dst,
        n
    ))

    distances = np.asarray(
        graph.distances(
            source=sources,
            target=targets,
            weights=weights if edges else None,
            mode="out"
        ),
        dtype=float
    )

    # Penaliteti per lidhjet qe nuk ekzistojne.
    distances[~np.isfinite(distances)] = 10000.0

    return float(np.mean(distances))

def load_vectors(path, samples):
    document = json.loads(
        path.read_text(encoding="utf-8")
    )

    payload = (
        document[0]
        if isinstance(document, list)
        else document
    )

    vectors = payload["decisionVector"]

    if len(vectors) != 100:
        raise RuntimeError(
            "Submission duhet te kete 100 konfigurime."
        )

    return vectors[:samples]

def evaluate_vectors(vectors):
    results = []

    start = time.perf_counter()

    for i, x in enumerate(vectors, 1):
        f = np.asarray(
            udp.fitness(x),
            dtype=float
        )

        results.append(f)

        print(
            f"Konfigurimi {i}/{len(vectors)}",
            flush=True
        )

    elapsed = time.perf_counter() - start

    return np.asarray(results), elapsed

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
        default=10
    )

    args = parser.parse_args()

    if not args.base.exists():
        parser.error(
            f"Mungon fajlli: {args.base}"
        )

    if not 1 <= args.samples <= 100:
        parser.error(
            "--samples duhet te jete 1 deri 100"
        )

    vectors = load_vectors(
        args.base,
        args.samples
    )

    original_method = udp.average_shortest_path

    print("=" * 55)
    print("ESA SpOC 2 - NETWORKX vs IGRAPH")
    print("=" * 55)

    # Testi i modelit origjinal.
    print("\n1. MODELI ORIGJINAL ESA")

    original, original_time = evaluate_vectors(
        vectors
    )

    # Testi i versionit te shpejte.
    print("\n2. MODELI IGRAPH")

    try:
        udp.average_shortest_path = MethodType(
            fast_average_shortest_path,
            udp
        )

        fast, fast_time = evaluate_vectors(
            vectors
        )

    finally:
        # Rikthe gjithmone funksionin origjinal.
        udp.average_shortest_path = original_method

    differences = np.abs(original - fast)

    max_difference = float(
        np.max(differences)
    )

    correct = bool(
        np.allclose(
            original,
            fast,
            rtol=0,
            atol=1e-8
        )
    )

    print("\n" + "=" * 55)
    print("REZULTATI PERFUNDIMTAR")
    print("=" * 55)

    print(
        f"Konfigurime: {len(vectors)}"
    )

    print(
        f"Koha origjinale: {original_time:.3f} sekonda"
    )

    print(
        f"Koha igraph: {fast_time:.3f} sekonda"
    )

    if fast_time > 0:
        print(
            f"Speedup: {original_time / fast_time:.2f}x"
        )

    print(
        f"Dallimi maksimal: {max_difference:.12g}"
    )

    print(
        f"Rezultatet perputhen: {correct}"
    )

    if not correct:
        print(
            "DESHTIM: Mos e perdor versionin e shpejte!"
        )

        print(
            "Dallimet sipas J1, J2, C1, C2:"
        )

        print(
            np.max(differences, axis=0)
        )

    elif fast_time < original_time:
        print(
            "SUKSES: igraph eshte me i shpejte "
            "dhe rezultatet perputhen."
        )

    else:
        print(
            "Rezultatet perputhen, por igraph "
            "nuk ishte me i shpejte ne kete test."
        )

if __name__ == "__main__":
    main()