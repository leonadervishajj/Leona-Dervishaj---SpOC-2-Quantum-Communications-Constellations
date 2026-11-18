
import argparse
import json
import time
from pathlib import Path
from types import MethodType

import networkx as nx
import numpy as np
from numba import njit


@njit(cache=True)
def _geometry_graph(xyz, n1, eta1, eta2, los_limit, eps_z, n_rovers):
    """Replicate ESA build_graph's pair order, LOS, zenith and QKD rules."""
    n = xyz.shape[0]
    adj = np.zeros((n, n), dtype=np.float64)
    d_min = np.inf

    for i in range(n):
        ax = xyz[i, 0]
        ay = xyz[i, 1]
        az = xyz[i, 2]

        na2 = ax * ax + ay * ay + az * az
        na = np.sqrt(na2)

        for j in range(i):
            bx = xyz[j, 0]
            by = xyz[j, 1]
            bz = xyz[j, 2]

            dx = bx - ax
            dy = by - ay
            dz = bz - az

            dist = np.sqrt(
                dx * dx + dy * dy + dz * dz
            )

            # ESA: line_of_sight(r1=xyz[i], r2=xyz[j]).
            if dist < 1e-6:
                los = na
                cos_z = 0.0
            else:
                h1 = (
                    ax * dx + ay * dy + az * dz
                ) / dist

                arg = na2 - h1 * h1

                los = (
                    np.sqrt(arg)
                    if arg > 1e-6
                    else 0.0
                )

                cos_z = (
                    ax * dx + ay * dy + az * dz
                ) / (dist * na)

            if los >= los_limit or cos_z > 0.0:
                eta_j = (
                    eta1 if j < n1 else eta2
                )

                # Original qkd_metric obtains link distance from src-dst.
                d_link = dist

                weight = (
                    -np.log(eta_j)
                    + 2.0 * np.log(d_link)
                )

                if weight < 0.0:
                    weight = 1e3

                # Original: qkd_metric(idx=n-i,...), idx<=n_rovers.
                if n - i <= n_rovers:
                    if cos_z >= eps_z:
                        weight += (
                            1.0
                            / np.sin(
                                np.pi / 2.0
                                - np.arccos(cos_z)
                            )
                        )
                    else:
                        weight = 0.0

                adj[i, j] = weight
                adj[j, i] = weight

                if d_link < d_min:
                    d_min = d_link

    return adj, d_min


def fast_build_graph(
    self, ep_idx, pos, num_w1_sats, eta
):
    """Drop-in replacement for udp.build_graph with identical return types."""

    xyz = np.ascontiguousarray(
        pos[:, ep_idx, :],
        dtype=np.float64
    )

    adj, d_min = _geometry_graph(
        xyz,
        int(num_w1_sats),
        float(eta[0]),
        float(eta[1]),
        float(self.LOS),
        float(self.eps_z),
        int(self.n_rovers),
    )

    return (
        nx.from_numpy_array(adj),
        adj,
        float(d_min)
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--base',
        required=True,
        type=Path
    )

    p.add_argument(
        '--samples',
        type=int,
        default=8
    )

    p.add_argument(
        '--out',
        type=Path,
        default=Path(
            'results/numba_geometry_test'
        )
    )

    p.add_argument(
        '--tol',
        type=float,
        default=1e-9
    )

    args = p.parse_args()

    if args.samples < 1:
        p.error('--samples duhet >= 1')

    from constellation import udp

    payload = json.loads(
        args.base.read_text(
            encoding='utf-8'
        )
    )

    payload = (
        payload[0]
        if isinstance(payload, list)
        else payload
    )

    vectors = payload[
        'decisionVector'
    ][:args.samples]

    if not vectors:
        p.error(
            'Nuk ka vektore per test'
        )

    original = udp.build_graph

    # JIT warm-up with a small, safe dummy graph.
    # Not included in timing.
    _geometry_graph(
        np.array(
            [
                [7000., 0., 0.],
                [0., 7000., 0.]
            ],
            dtype=np.float64
        ),
        1,
        20.,
        20.,
        float(udp.LOS),
        float(udp.eps_z),
        int(udp.n_rovers)
    )

    try:
        t0 = time.perf_counter()

        baseline = [
            np.asarray(
                udp.fitness(x),
                dtype=float
            )
            for x in vectors
        ]

        original_seconds = (
            time.perf_counter() - t0
        )

        udp.build_graph = MethodType(
            fast_build_graph,
            udp
        )

        t0 = time.perf_counter()

        new = [
            np.asarray(
                udp.fitness(x),
                dtype=float
            )
            for x in vectors
        ]

        fast_seconds = (
            time.perf_counter() - t0
        )

    finally:
        udp.build_graph = original

    max_error = max(
        float(
            np.max(
                np.abs(a - b)
            )
        )
        for a, b in zip(
            baseline, new
        )
    )

    matches = all(
        np.allclose(
            a,
            b,
            rtol=args.tol,
            atol=args.tol
        )
        for a, b in zip(
            baseline, new
        )
    )

    result = {
        'samples': len(vectors),
        'original_seconds': original_seconds,
        'numba_seconds': fast_seconds,
        'speedup': (
            original_seconds / fast_seconds
        ),
        'max_abs_difference': max_error,
        'results_match': bool(matches),
    }

    args.out.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        args.out / 'benchmark.json'
    ).write_text(
        json.dumps(
            result,
            indent=2
        ),
        encoding='utf-8'
    )

    print(
        'BENCHMARK RESULTS:',
        json.dumps(
            result,
            indent=2
        ),
        flush=True
    )

    if not matches:
        raise SystemExit(
            'STOP: rezultatet nuk perputhen '
            'me ESA, mos perdor fast_build_graph!'
        )

    print(
        'PASS: Numba build_graph perputhet '
        'me ESA ne mostren e testuar.',
        flush=True
    )


if __name__ == '__main__':
    main()
