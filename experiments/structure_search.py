
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

REF = np.array([1.2, 1.4])
CHALLENGE = "spoc-2-quantum-communications-constellations"
PROBLEM = "quantum-communications-constellations"
S_RANGE, P_RANGE, F_RANGE = (4, 10), (2, 10), (0, 9)

_udp = None


def get_udp():
    """Ngarkon UDP-në e ESA-s një herë për proces."""
    global _udp
    if _udp is None:
        from constellation import udp
        _udp = udp
    return _udp


# --------------------------------------------------------------------------
# Pareto / hipervëllim (2D, pa pygmo)
# --------------------------------------------------------------------------
def nondominated(F):
    """Indekset e pikave jo-të-dominuara, të renditura sipas J1 rritës."""
    F = np.asarray(F, dtype=float)
    order = np.lexsort((F[:, 1], F[:, 0]))
    keep, best = [], np.inf
    for i in order:
        if F[i, 1] < best:
            keep.append(int(i))
            best = F[i, 1]
    return keep


def hv2d(F):
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        return 0.0
    F = F[(F[:, 0] < REF[0]) & (F[:, 1] < REF[1])]
    if len(F) == 0:
        return 0.0
    F = F[nondominated(F)]
    hv, prev2 = 0.0, REF[1]
    for f1, f2 in F:
        hv += (REF[0] - f1) * (prev2 - f2)
        prev2 = f2
    return float(hv)


def select_k(X, F, k=100):
    """Mban k zgjidhje me HV maksimal (heq pikën me kontributin më të vogël)."""
    F = np.asarray(F, dtype=float)[:, :2]
    idx = nondominated(F)
    while len(idx) > k:
        f = F[idx]
        right1 = np.append(f[1:, 0], REF[0])
        up2 = np.insert(f[:-1, 1], 0, REF[1])
        contrib = (right1 - f[:, 0]) * (up2 - f[:, 1])
        idx.pop(int(np.argmin(contrib)))
    return [X[i] for i in idx], F[idx]


# --------------------------------------------------------------------------
# Arkivi
# --------------------------------------------------------------------------
def _feasible(x, f):
    """f = [J1, J2] (checkpoint-et e ruajnë vetëm objektivat, popullata e tyre është e lejueshme)
    ose [J1, J2, C1, C2] (atëherë kontrollojmë kufizimet)."""
    if len(x) != 20 or len(f) < 2 or not (f[0] < REF[0] and f[1] < REF[1]):
        return False
    return len(f) < 4 or (f[2] <= 0 and f[3] <= 0)


def load_archive(results_dir):
    """Lexon popullatat nga çdo checkpoint.json + arkivin e vet."""
    X, F, seen = [], [], set()

    def add(x, f):
        if not _feasible(x, f):
            return
        key = tuple(round(float(v), 9) for v in x)
        if key in seen:
            return
        seen.add(key)
        X.append([float(v) for v in x])
        F.append([float(f[0]), float(f[1])])

    for p in sorted(Path(results_dir).rglob("checkpoint.json")):
        try:
            c = json.load(open(p))
        except Exception:
            continue
        for m in c.get("population", []):
            if isinstance(m, dict) and "x" in m and "f" in m:
                add(m["x"], m["f"])
    own = Path(results_dir) / "structure_search" / "archive.json"
    if own.exists():
        d = json.load(open(own))
        for x, f in zip(d["X"], d["F"]):
            add(x, f)
    return X, F


def reduce_archive(X, F):
    keep = nondominated(np.asarray(F)[:, :2])
    return [X[i] for i in keep], [F[i] for i in keep]


def save_archive(results_dir, X, F):
    d = Path(results_dir) / "structure_search"
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"X": X, "F": F}, open(d / "archive.json", "w"))


# --------------------------------------------------------------------------
# Struktura diskrete
# --------------------------------------------------------------------------
def rover_ok(r):
    u = get_udp()
    idx = np.array(r, dtype=int)
    c, _ = u.get_rover_constraint(u.lambdas[idx], u.phis[idx])
    return c <= 0


def neighbor(ints, rng):
    """Fqinj diskret: S/P/F ±1 ose një rover tjetër (me kontroll 3000 km)."""
    g = [int(v) for v in ints]
    ranges = [S_RANGE, P_RANGE, F_RANGE, S_RANGE, P_RANGE, F_RANGE]
    for _ in range(int(rng.choice([1, 1, 1, 2]))):
        if rng.random() < 0.7:
            i = int(rng.integers(0, 6))
            lo, hi = ranges[i]
            g[i] = int(np.clip(g[i] + int(rng.choice([-1, 1])), lo, hi))
        else:
            for _try in range(30):
                j = 6 + int(rng.integers(0, 4))
                old = g[j]
                g[j] = int(rng.integers(0, 100))
                if len(set(g[6:])) == 4 and rover_ok(g[6:]):
                    break
                g[j] = old
    return g


# --------------------------------------------------------------------------
# Detyra e një punëtori: optimizo 10 variablat e vazhdueshëm për një strukturë
# --------------------------------------------------------------------------
def run_task(task):
    ints, w, x0, popsize, maxiter, seed = task
    u = get_udp()
    if not rover_ok(ints[6:]):
        return [], []
    lb, ub = u.get_bounds()
    bounds = list(zip(lb[:10], ub[:10]))
    x0 = np.clip(np.asarray(x0, dtype=float), lb[:10], ub[:10])
    found = []

    def obj(c):
        x = [float(v) for v in c] + [int(v) for v in ints]
        f = [float(v) for v in u.fitness(x)]
        feas = f[2] <= 0 and f[3] <= 0
        if feas and f[0] < REF[0] and f[1] < REF[1]:
            found.append((x[:10] + [float(v) for v in ints], f[:2]))
        n1 = (f[0] - 0.70) / (REF[0] - 0.70)
        n2 = f[1] / REF[1]
        g = max(w * n1, (1 - w) * n2) + 0.01 * (n1 + n2)   # Tchebycheff i shtuar
        if not feas:
            g += 10 + max(0.0, f[2]) / 1000 + max(0.0, f[3]) / 1000
        return g

    try:
        differential_evolution(obj, bounds, maxiter=maxiter, popsize=popsize,
                               seed=seed, x0=x0, polish=False, tol=0,
                               updating="immediate")
    except Exception as e:  # një detyrë e dështuar nuk duhet ta ndalë gjithçka
        print("task error:", e)
    if not found:
        return [], []
    keep = nondominated(np.array([f[:2] for _, f in found]))
    return [found[i][0] for i in keep], [found[i][1] for i in keep]


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--output", default="submission/esa_submission_structure.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8, help="detyra për raund")
    ap.add_argument("--popsize", type=int, default=6, help="DE popsize (x10 variabla)")
    ap.add_argument("--maxiter", type=int, default=12, help="DE brezat për detyrë")
    ap.add_argument("--p-keep", type=float, default=0.3,
                    help="probabiliteti të ruhet struktura (vetëm pesha ndryshon)")
    ap.add_argument("--time-limit", type=float, default=None, help="minuta")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    X, F = load_archive(args.results)
    if not X:
        raise SystemExit("Arkivi bosh: nuk u gjet asnjë checkpoint.json me 'population'.")
    X, F = reduce_archive(X, F)
    print(f"Arkivi fillestar: {len(X)} pika jo-të-dominuara, "
          f"HV = {hv2d(np.array(F)[:, :2]):.6f}")

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for rnd in range(1, args.rounds + 1):
            tasks = []
            for _ in range(args.batch):
                px = X[int(rng.integers(0, len(X)))]
                ints = [int(round(v)) for v in px[10:]]
                if rng.random() >= args.p_keep:
                    ints = neighbor(ints, rng)
                w = float(rng.uniform(0.05, 0.95))
                tasks.append((ints, w, px[:10], args.popsize, args.maxiter,
                              int(rng.integers(0, 2**31 - 1))))
            for xs, fs in pool.imap_unordered(run_task, tasks):
                X += xs
                F += fs
            X, F = reduce_archive(X, F)
            hv = hv2d(np.array(F)[:, :2])
            save_archive(args.results, X, F)
            print(f"Raundi {rnd}/{args.rounds}: arkivi {len(X)} pika, "
                  f"HV = {hv:.6f}, rezultati ≈ {-hv * 1e4:.2f}, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if args.time_limit and (time.time() - t0) / 60 > args.time_limit:
                print("Kufiri i kohës u arrit.")
                break

    Xs, Fs = select_k(X, F, 100)
    if not args.no_verify:                       # rivlerëso me fitness-in e ESA-s
        u = get_udp()
        Fv = [u.fitness([float(v) for v in x[:10]] + [int(v) for v in x[10:]]) for x in Xs]
        bad = [i for i, f in enumerate(Fv) if not _feasible(Xs[i], f)]
        if bad:
            raise SystemExit(f"{len(bad)} zgjidhje nuk kalojnë verifikimin; nuk po ruaj.")
        Fs = np.array([[f[0], f[1]] for f in Fv])  # J1, J2
        print("Verifikimi: të gjitha zgjidhjet janë të lejueshme.")
    hv = hv2d(Fs)
    from submission_helper import create_submission
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    create_submission(CHALLENGE, PROBLEM, Xs, fn_out=args.output,
                      name="structure_search",
                      description="Memetic search: discrete neighbourhood + DE on continuous genes")
    print(f"U ruajtën {len(Xs)} zgjidhje në {args.output}: HV = {hv:.6f}, "
          f"rezultati ≈ {-hv * 1e4:.2f}")


if __name__ == "__main__":
    main()