"""One route on one CPU. Reuse the archived geometry builder, loss and training loop."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import pickle
import platform
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def clocks():
    return time.perf_counter(), time.process_time()


def elapsed(start):
    return {'wall_seconds': time.perf_counter() - start[0], 'cpu_seconds': time.process_time() - start[1]}


def prepare_inputs(key, targets):
    """Upstream only. Match every requested training array against its original hash."""
    import numpy as np
    from physical128_data import array_hash
    from proximal_spo.feasible_policies import _geometry_cache_key
    from scaled_data_neural_data import generate_extension
    from simplified_data import load_structure

    base = ROOT / 'base' / key
    bank = ROOT / 'frozen_data' / key
    if (bank / 'receipt.json').exists():
        return
    bank.mkdir(parents=True, exist_ok=True)
    for name in ('structure.npz', 'structure.json'):
        (bank / name).write_bytes((base / name).read_bytes())
    with np.load(base / 'observations.npz', allow_pickle=False) as archive:
        arrays = {f: archive['train__' + f] for f in ('contexts', 'physical_costs')}
    maximum = max(map(int, targets))
    route, seed = key.split('__data')
    if maximum > 100:
        extension, physical_model, _ = generate_extension(route, int(seed), 900)
        if _geometry_cache_key(physical_model) != _geometry_cache_key(load_structure(base)):
            raise ValueError('Additional observations use different constraints')
        arrays = {f: np.concatenate([v, extension[f]]) for f, v in arrays.items()}
    for value, target in targets.items():
        n = int(value)
        for field, fingerprint in [('contexts', 'prefix_context_sha256'), ('physical_costs', 'prefix_cost_sha256')]:
            if array_hash(arrays[field][:n]) != target[fingerprint]:
                raise ValueError(f'Observed {field} differs from the original {key} N={n}')
    hashes = {}
    for n in [size for size in (100, 500, 1000) if size <= maximum]:
        path = bank / f'observations_n{n}.npz'
        with path.open('wb') as stream:
            np.savez_compressed(stream, **{'train__' + k: v[:n] for k, v in arrays.items()})
        hashes[path.name] = sha(path)
    write(bank / 'receipt.json', {'key': key, 'files_sha256': hashes, 'verified_training_counts': list(targets),
                                  'inputs': 'original observed prefix plus archived extension seed and generator',
                                  'original_observations_file_sha256': sha(base / 'observations.npz')})


def verified_geometry(key, maximum, manifest):
    import bank_geometry as builder
    from simplified_data import load_structure

    seed = manifest['seeds']['initialization']
    where = builder.directory(key, seed)
    target = manifest['banks'][key]
    prep_start = clocks()
    prepare_inputs(key, target)
    upstream = elapsed(prep_start)
    started = clocks()
    cache_existed = (where / f'prefix{maximum}.json').exists()
    # The original continuation replays the solver order and restores per-sample RNG checkpoints.
    if not cache_existed:
        builder.prepare(ROOT, key, maximum=maximum, initialization_seed=seed, reuse_random_free=False)
    geometry_build = elapsed(started)
    started = clocks()
    for n, saved in target.items():
        if int(n) > maximum:
            continue
        prefix = read(where / f'prefix{n}.json')
        for field, expected in saved.items():
            if prefix[field] != expected:
                raise ValueError(f'Rebuilt geometry differs: {key}, N={n}, {field}')
        if sha(where / f'prefix{n}.pickle') != prefix['metadata_sha256']:
            raise ValueError(f'Geometry index differs for {key}, N={n}')
    # The two historically random-free banks had no stored per-vertex hashes.
    # Each other bank retained a digest for each unique original anchor.
    vertex_digests = manifest['original_vertex_hashes'][key]
    checked = 0
    if vertex_digests:
        # This is the new builder's prefix whose bytes were checked above against the archive.
        with (where / f'prefix{maximum}.pickle').open('rb') as stream:
            names = set(pickle.load(stream)['vertices'])
        for name in names:
            if name not in vertex_digests or vertex_digests[name] != sha(where / name):
                raise ValueError(f'Original adjacent vertices differ: {key}/{name}')
            checked += 1
        if checked != read(where / f'prefix{maximum}.json')['unique_anchor_cache_files']:
            raise ValueError('Verified anchor count differs')
    attempt = ROOT / 'preprocessing' / key / f'{time.time_ns()}.json'
    write(attempt, {'N': maximum, 'already_prepared': cache_existed, 'observations': upstream,
                    'geometry_construction': geometry_build, 'identity_checks': elapsed(started),
                    'original_vertices_checked': checked, 'training_included': False})
    return load_structure(ROOT / 'frozen_data' / key), str(attempt.relative_to(ROOT))


def materialize(lazy):
    """Preserve raw arrays, dtypes, row order and public collation; open each unique file once."""
    import numpy as np
    resident = copy.copy(lazy)
    for name in ('ctrs', 'adjacent_verts'):
        sequence = getattr(lazy, name)
        by_file = {}
        rows = []
        for i, filename in enumerate(sequence.names):
            if filename not in by_file:
                by_file[filename] = np.array(sequence[i], copy=True, order='K')
            rows.append(by_file[filename])
        setattr(resident, name, rows)
    return resident


def check_parity(job, contexts, costs, lazy, resident):
    import numpy as np
    import torch
    import run_public_solver_free_matched as matched
    from proximal_spo.public_highs import public_geometry
    from proximal_spo.seeds import derive_seed
    from simplified_losses import public_loss, batch_loss

    config = job['config']
    seeds = config['seeds']
    matched._seed(seeds['initialization'])
    model = matched._make_model(config, contexts, costs)
    loss = public_loss('lava', lazy.model, job['parameters'], lazy, seeds['perturbation'], weighted=False)
    collate = public_geometry(str(matched.PUBLIC_REPO)).collate_fn_2
    order = np.random.default_rng(derive_seed(seeds['shuffle'], 'epoch', 1)).permutation(len(contexts))
    size = config['training']['batch_size']
    batches = [order[i:i+size] for i in range(0, len(order), size)]
    cases = []
    nonzero = False
    for position in sorted({0, len(batches)//2 - 1, len(batches) - 2, len(batches) - 1}):
        idx = batches[position]
        old = collate([lazy[int(i)] for i in idx])
        new = collate([resident[int(i)] for i in idx])
        if len(old) != len(new) or any(a.dtype != b.dtype or not torch.equal(a, b) for a, b in zip(old, new)):
            raise ValueError(f'Input tensors/padding differ at batch {position}')

        def evaluate(batch):
            prediction = matched._physical_prediction(model, torch.as_tensor(contexts[idx], dtype=torch.float32))
            value = batch_loss('lava', loss, prediction, batch)
            grads = torch.autograd.grad(value, (prediction, *model.parameters()))
            return [value.detach(), *(g.detach() for g in grads)]

        a, b = evaluate(old), evaluate(new)
        if any(not torch.isfinite(x).all() or not torch.equal(x, y) for x, y in zip(a, b)):
            raise ValueError(f'LAVA loss or gradients differ at batch {position}')
        nonzero |= bool(torch.count_nonzero(a[1]))
        cases.append({'position': position, 'indices': idx.tolist(), 'loss': float(a[0]),
                      'prediction_gradient_norm': float(a[1].norm()),
                      'parameter_gradient_norms': [float(g.norm()) for g in a[2:]]})
        del old, new
    if not nonzero:
        raise ValueError('Parity check did not exercise a nonzero prediction gradient')
    return {'status': 'passed', 'original_fit_id': job['id'], 'seeds': seeds,
            'comparison': 'exact equality of all collated tensors, loss, prediction and all parameter gradients',
            'optimizer_updates': 0, 'epoch_order': 1, 'batches': cases}


def completed_receipt(item):
    path = ROOT / 'results' / (item['job']['id'] + '.json')
    if not path.is_file():
        return False
    value = read(path)
    if (value.get('status'), value.get('original_receipt_sha256'), value.get('epochs'), value.get('test_access')) != (
            'complete', item['original_receipt_sha256'], 300, False):
        raise ValueError(f'Conflicting timing receipt {path}')
    return True


def first_passed(manifest):
    path = ROOT / 'parity.json'
    if not path.is_file():
        return False
    gate = read(path)
    return (gate.get('status') == 'passed' and gate.get('original_fit_id') == manifest['first']
            and gate.get('manifest_sha256') == sha(ROOT / 'manifest.json')
            and completed_receipt(manifest['jobs'][manifest['first']]))


def time_winner(item, problem, geometry_record, manifest):
    import numpy as np
    import torch
    import bank_geometry
    import final_geometry_cache
    import run_central_path_final as maintained

    original = item['job']
    first = original['id'] == manifest['first']
    if not first and not first_passed(manifest):
        raise RuntimeError('SP25 N500 parity and complete timing are required')
    if completed_receipt(item):
        return
    started = clocks()
    job = copy.deepcopy(original)
    config = job['config']
    job.update(stage='timing', frozen_horizon=300)
    config['evaluate_test'] = False
    config['training']['selection_mode'] = 'last'
    maintained.validate_timing_job(job)
    n = config['sizes']['train']
    stored = 100 if n == 100 else 500 if n <= 500 else 1000
    key = job['required_data_key']
    with np.load(ROOT / 'frozen_data' / key / f'observations_n{stored}.npz', allow_pickle=False) as arr:
        contexts = arr['train__contexts'][:n].copy()
        costs = arr['train__physical_costs'][:n].copy()
    loaded = SimpleNamespace(problem=problem, dataset=SimpleNamespace(train=SimpleNamespace(contexts=contexts, physical_costs=costs)))
    attempt = ROOT / 'attempts' / original['id'] / str(time.time_ns())
    attempt.mkdir(parents=True)
    setup = {}

    def update(phase, **fields):
        if phase != 'training' or fields.get('_force_progress') or fields.get('epochs_completed') == 1:
            write(attempt / 'progress.json', {'phase': phase, 'time': time.time(), **fields})

    def geometry(job, solver, x, y, notify):
        baseline = bank_geometry.get_geometry(job, solver, x, y, notify, initialization_seed=config['seeds']['initialization'])
        opened = clocks()
        new = materialize(baseline)
        setup['one_time_loader'] = elapsed(opened)
        if first:
            opened = clocks()
            gate = check_parity(original, x, y, baseline, new)
            gate.update(manifest_sha256=sha(ROOT / 'manifest.json'), slurm_job_id=os.environ['SLURM_JOB_ID'],
                        host=platform.node())
            write(ROOT / 'parity.json', gate)
            setup['parity'] = elapsed(opened)
        return new

    def optimizer_ready():
        setup['before_optimizer_including_parity'] = elapsed(started)
        update('optimizer_ready')

    previous = final_geometry_cache.get_geometry
    final_geometry_cache.get_geometry = geometry
    try:
        summary, _ = maintained.baseline_fit(job, loaded, attempt, update, optimizer_ready, timing_only=True)
    finally:
        final_geometry_cache.get_geometry = previous
    if (summary['epochs_completed'], summary['optimization_cap_reached'], summary['validation_wall_seconds'],
        summary['test'], summary['optimizer']) != (300, False, 0., None, 'Adam'):
        raise RuntimeError('Incomplete training clock or evaluation entered timing replay')
    write(ROOT / 'results' / (original['id'] + '.json'), {
        'status': 'complete', 'original_fit_id': original['id'], 'original_receipt_sha256': item['original_receipt_sha256'],
        'route': job['route'], 'N': n, 'depth': job['architecture_depth'], 'seeds': config['seeds'], 'epochs': 300,
        'training_wall_seconds': summary['optimization_wall_seconds'], 'training_cpu_seconds': summary['optimization_cpu_seconds'],
        'geometry_preparation_record': geometry_record, 'setup': setup, 'test_access': False, 'validation_wall_seconds': 0.,
        'regret_source': 'unchanged original confirmation', 'parameter_sha256_at_epoch300': summary['parameter_sha256'],
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'host': platform.node(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'torch': str(torch.__version__), 'torch_threads': torch.get_num_threads(), 'numpy': np.__version__})
    gc.collect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('key')
    p.add_argument('phase', choices=('first', 'remaining'))
    args = p.parse_args()
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_NODELIST')
            or len(os.sched_getaffinity(0)) != 1 or platform.node().split('.')[0] == 'master01-hpc'
            or Path(os.environ['LAVA_RUN_ROOT']).resolve() != ROOT
            or Path(os.environ['LAVA_GEOMETRY_ROOT']).resolve() != Path('/tmp/lava_timing_replay_20261107')):
        raise RuntimeError('Expected one bound Slurm CPU, compute node and isolated node-local geometry')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = read(ROOT / 'manifest.json')
    if args.phase == 'remaining' and not first_passed(manifest):
        raise RuntimeError('Initial SP25 fit has not passed')
    items = [i for k,i in manifest['jobs'].items() if i['job']['required_data_key'] == args.key
             and (k == manifest['first']) == (args.phase == 'first') and not completed_receipt(i)]
    if not items:
        return
    items.sort(key=lambda i: (i['job']['config']['sizes']['train'], i['job']['architecture_depth']))
    maximum = max(i['job']['config']['sizes']['train'] for i in items)
    problem, record = verified_geometry(args.key, maximum, manifest)
    for item in items:
        time_winner(item, problem, record, manifest)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write(ROOT / 'failures' / f'{time.time_ns()}-{os.getpid()}.json',
              {'arguments': sys.argv[1:], 'error': traceback.format_exc(), 'time': time.time()})
        raise
