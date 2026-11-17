
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / 'submission' / 'esa_submission_structure.json'
DEFAULT_OUT = ROOT / 'results' / 'global_structure_search'
REF = (1.2, 1.4)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def clean_vector(x, lb, ub):
    a = np.clip(np.asarray(x, dtype=float), lb, ub)
    a[10:] = np.rint(a[10:])
    return a.tolist()


def key(x):
    return tuple(round(float(v), 10) for v in x)


def hv2d(items):
    """Exact 2D minimization hypervolume against (1.2, 1.4)."""
    points = sorted((float(z['f'][0]), float(z['f'][1]))
                    for z in items if z['f'][0] < REF[0] and z['f'][1] < REF[1])
    best_y, area = REF[1], 0.0
    for x, y in points:
        if y < best_y:
            area += (REF[0] - x) * (best_y - y)
            best_y = y
    return float(area)


def select100(items):
    """Remove smallest exact HV contributor until 100 remain."""
    selected = list(items)
    while len(selected) > 100:
        total = hv2d(selected)
        losses = [total - hv2d(selected[:i] + selected[i + 1:])
                  for i in range(len(selected))]
        selected.pop(int(np.argmin(losses)))
    return selected


def insert_if_better(best, candidate, current_hv):
    if key(candidate['x']) in {key(z['x']) for z in best}:
        return best, current_hv, False
    trial = select100(best + [candidate])
    new_hv = hv2d(trial)
    if new_hv > current_hv + 1e-12:
        return trial, new_hv, True
    return best, current_hv, False


def new_proposal(rng, champions, island, lb, ub):
    """Large-scale structured exploration; not just +/-1 local mutation."""
    pool = champions + island
    seed = np.array(pool[int(rng.integers(len(pool)))]['x'], dtype=float)
    mode = int(rng.integers(5))
    x = seed.copy()

    if mode == 0:  # FULL STRUCTURE RESTART: change BOTH Walker structures
        x[:10] = rng.uniform(lb[:10], ub[:10])
        for ix in range(10, 16):
            x[ix] = rng.integers(int(lb[ix]), int(ub[ix]) + 1)
        if rng.random() < .65:
            x[16:] = rng.choice(100, size=4, replace=False)

    elif mode == 1:  # MIXED RESTART: new S/P/F and completely new rover positions
        for ix in (10, 11, 12, 13, 14, 15):
            x[ix] = rng.integers(int(lb[ix]), int(ub[ix]) + 1)
        x[16:] = rng.choice(100, size=4, replace=False)
        for ix in rng.choice(10, size=int(rng.integers(3, 11)), replace=False):
            x[ix] = rng.uniform(lb[ix], ub[ix])

    elif mode == 2:  # ISLAND DE: DE/rand/1 over continuous dimensions
        if len(pool) >= 3:
            idx = rng.choice(len(pool), size=3, replace=False)
            a, b, c = [np.asarray(pool[int(i)]['x'], dtype=float) for i in idx]
            F = rng.uniform(.4, 1.1)
            mutant = a[:10] + F * (b[:10] - c[:10])
            mask = rng.random(10) < .85
            mask[int(rng.integers(10))] = True
            x[:10] = np.where(mask, mutant, x[:10])
        else:
            x[:10] = rng.uniform(lb[:10], ub[:10])
        for ix in rng.choice(np.arange(10, 16), size=int(rng.integers(2, 7)), replace=False):
            x[ix] = rng.integers(int(lb[ix]), int(ub[ix]) + 1)
        for ix in rng.choice(np.arange(16, 20), size=int(rng.integers(1, 5)), replace=False):
            x[ix] = rng.integers(0, 100)

    elif mode == 3:  # RECOMBINATION: radically different Walker + rover sources
        partner = np.array(pool[int(rng.integers(len(pool)))]['x'], dtype=float)
        if rng.random() < .5:
            x[:5] = partner[:5]
            x[10:13] = partner[10:13]
        else:
            x[5:10] = partner[5:10]
            x[13:16] = partner[13:16]
        x[:10] += rng.normal(0, .22, 10) * (ub[:10] - lb[:10])
        x[16:] = rng.choice(100, size=4, replace=False)

    else:  # WIDE JUMPS: change an entire Walker or rovers, not tiny steps
        walker = int(rng.integers(2))
        j = slice(5 * walker, 5 * walker + 5)
        x[j] = rng.uniform(lb[j], ub[j])
        k = slice(10 + 3 * walker, 13 + 3 * walker)
        for ix in range(k.start, k.stop):
            x[ix] = rng.integers(int(lb[ix]), int(ub[ix]) + 1)
        if rng.random() < .8:
            x[16:] = rng.choice(100, size=4, replace=False)
    return clean_vector(x, lb, ub), mode


def run(args):
    # Module invocation resolves constellation.py at the repository root.
    from constellation import udp

    rng = np.random.default_rng(args.seed)
    lb, ub = (np.asarray(t, dtype=float) for t in udp.get_bounds())
    base_doc = json.loads(args.base.read_text(encoding='utf-8'))
    payload = base_doc[0] if isinstance(base_doc, list) else base_doc
    vectors = payload['decisionVector']
    if len(vectors) != 100:
        raise ValueError('Baseline must contain exactly 100 decision vectors.')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'checkpoint.json'
    submission = out / 'esa_submission_global_best.json'

    print('Verifikimi i rekordit ekzistues: 100 fitness ESA...', flush=True)
    champions = []
    for i, x in enumerate(vectors):
        f = np.asarray(udp.fitness(x), dtype=float)
        if f.shape[0] < 4 or not np.all(np.isfinite(f)) or np.any(f[2:] > 0):
            raise RuntimeError(f'Baseline invalid at index {i}: {f}')
        champions.append({'x': clean_vector(x, lb, ub), 'f': f[:2].tolist()})
    best = champions[:]
    baseline_hv = hv2d(best)
    current_hv = baseline_hv
    print(f'BASELINE VERIFIED: {baseline_hv:.12f} | feasible 100/100', flush=True)

    # Resume safely: re-verify an earlier best generated by THIS script.
    # Keep the original benchmark unchanged; never downgrade an earlier record.
    if submission.exists():
        print('Previous global result found: verifying before resume...', flush=True)
        saved_doc = json.loads(submission.read_text(encoding='utf-8'))
        saved_payload = saved_doc[0] if isinstance(saved_doc, list) else saved_doc
        old_xs = saved_payload['decisionVector']
        if len(old_xs) != 100:
            raise RuntimeError('Previous saved result must have 100 vectors.')
        old_best = []
        for i, x in enumerate(old_xs):
            f = np.asarray(udp.fitness(x), dtype=float)
            if f.shape[0] < 4 or not np.all(np.isfinite(f)) or np.any(f[2:] > 0):
                raise RuntimeError(f'Previous saved result invalid #{i}: {f}')
            old_best.append({'x': clean_vector(x, lb, ub), 'f': f[:2].tolist()})
        old_hv = hv2d(old_best)
        if old_hv > current_hv + 1e-12:
            best, current_hv = old_best, old_hv
            print(f'RESUMED BEST: HV={current_hv:.12f}', flush=True)

    # Multiple independent search pools; protect global best separately.
    source_pool = champions + best
    islands = [[copy.deepcopy(source_pool[int(i)])
                for i in rng.choice(len(source_pool), size=12, replace=False)]
               for _ in range(args.islands)]
    seen = {key(z['x']) for z in champions}
    attempts = valid = accepted = invalid = errors = 0
    started = time.monotonic()

    def save():
        atomic_json(checkpoint, {'baseline_hv': baseline_hv, 'best_hv': current_hv,
                                 'attempted': attempts, 'valid': valid,
                                 'accepted': accepted, 'invalid': invalid,
                                 'errors': errors, 'population': best})
        # Preserve original metadata and exact 100-vector official JSON format.
        output_doc = copy.deepcopy(base_doc)
        target = output_doc[0] if isinstance(output_doc, list) else output_doc
        target['decisionVector'] = [z['x'] for z in best]
        atomic_json(submission, output_doc)

    save()  # Entire validated starting set safely stored from the beginning.
    for step in range(args.evaluations):
        if args.minutes and (time.monotonic() - started) >= args.minutes * 60:
            break
        island_idx = step % args.islands
        x, mode = new_proposal(rng, best, islands[island_idx], lb, ub)
        if key(x) in seen:
            continue
        seen.add(key(x))
        attempts += 1
        try:
            f = np.asarray(udp.fitness(x), dtype=float)
            if f.shape[0] < 4 or not np.all(np.isfinite(f)):
                invalid += 1
                continue
            if np.any(f[2:] > 0):
                invalid += 1
                continue
            valid += 1
            z = {'x': x, 'f': f[:2].tolist()}
            # Keep a diverse pool even if candidate does not improve HV directly.
            pool = islands[island_idx]
            pool.append(z)
            if len(pool) > args.pool:
                # Retain 1/2 best weighted objective and 1/2 random diverse solutions.
                weight = float(rng.uniform(.05, .95))
                ordered = sorted(pool, key=lambda a: weight*a['f'][0]/REF[0]
                                 + (1-weight)*a['f'][1]/REF[1])
                keep = ordered[:max(4, args.pool // 2)]
                rest = [a for a in pool if id(a) not in {id(q) for q in keep}]
                count = max(0, args.pool - len(keep))
                if count and rest:
                    keep.extend([rest[int(i)] for i in rng.choice(len(rest),
                                size=min(count, len(rest)), replace=False)])
                islands[island_idx] = keep
            updated, next_hv, improved = insert_if_better(best, z, current_hv)
            if improved:
                best, current_hv = updated, next_hv
                accepted += 1
                print(f'IMPROVED #{accepted}: HV={current_hv:.12f} '
                      f'(+{current_hv-baseline_hv:.9f}), step={step+1}, mode={mode}', flush=True)
                save()
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f'Fitness error at step {step+1}: {type(exc).__name__}: {exc}', flush=True)

        if (step+1) % args.report == 0:
            print(f'Step {step+1}/{args.evaluations} | valid={valid} | '
                  f'invalid={invalid} | accepted={accepted} | HV={current_hv:.12f}', flush=True)
            save()

    # Re-evaluate ALL 100 selected points using the unmodified ESA fitness.
    print('FINAL ESA VERIFICATION...', flush=True)
    validated = []
    for i, z in enumerate(best):
        f = np.asarray(udp.fitness(z['x']), dtype=float)
        if f.shape[0] < 4 or not np.all(np.isfinite(f)) or np.any(f[2:] > 0):
            raise RuntimeError(f'Invalid candidate #{i}: {f}')
        validated.append({'x': z['x'], 'f': f[:2].tolist()})
    verified_hv = hv2d(validated)
    if verified_hv + 1e-9 < baseline_hv:
        raise RuntimeError('Verified final HV lower than original baseline. Review output.')
    if abs(verified_hv - current_hv) > 1e-8:
        raise RuntimeError(f'HV changed after ESA re-evaluation: {current_hv} -> {verified_hv}')
    best, current_hv = validated, verified_hv
    save()
    print(f'VERIFIED: 100/100 | HV={verified_hv:.12f} | '
          f'GAIN={verified_hv-baseline_hv:.12f} | accepted={accepted} | '
          f'errors={errors}', flush=True)
    print('FILE:', submission, flush=True)


def cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, default=DEFAULT_BASE)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--evaluations', type=int, default=120)
    ap.add_argument('--minutes', type=float, default=0,
                    help='Optional time cap in minutes (0=off)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--islands', type=int, default=4)
    ap.add_argument('--pool', type=int, default=32)
    ap.add_argument('--report', type=int, default=10)
    a = ap.parse_args()
    if a.evaluations < 0 or a.islands < 1 or a.pool < 8 or a.report < 1:
        ap.error('Invalid search budget/pool/islands/report')
    if not a.base.exists():
        ap.error(f'Baseline submission not found: {a.base}')
    run(a)


if __name__ == '__main__':
    cli()
