"""Pinned benchmark inputs and resumable complete-bijection experiments.

Only the nine mechanisms explicitly discussed with the user are included.
Construction uses both sides to standardize offline simulation inputs. Privacy
claims apply to these standardized inputs, never to raw preprocessing.
"""
from __future__ import annotations

from functools import lru_cache
import csv
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import socket
import struct
import time

import numpy as np
import scipy

from .integral_mechanisms import Geometry, bound, evaluate_draw, solve, unit_channels
from .integral_study import atomic_json, save_npz, seed
from .login_guard import refuse_login_compute

NAME = 'clustering_benchmark_v1'
COMMIT = '4199c2fb21e39bab604106f8589480e0ed758df6'
DATASETS = {'wut/circles': (4000, 2), 'fcps/chainlink': (1000, 3),
            'graves/ring_noisy': (1050, 2), 'sipu/worms_64': (105000, 64),
            'uci/wdbc': (569, 30), 'mnist/digits10k': (10000, 784)}
# Explicit requested list, deliberately not populated from a mechanism registry.
ARMS = {1: ('gumbel', 'terminal_fold', 'anchor', 'ambient_radial', 'ambient_iid', 'cost_infinity', 'cost_iid'),
        2: ('gumbel', 'span_radial', 'coordinate_allocated', 'ambient_radial', 'ambient_iid', 'cost_infinity', 'cost_iid')}
NORMALIZATION = 'actual_QR_pooled_center_and_RMS_v1'
SEED_NAMESPACE = 'clustering-benchmark-20261112-v1'
DIMENSIONS = (784, 392, 256, 196, 98, 49, 24, 12, 6, 3)
SIZES = (16, 32, 64, 128, 256)
FLOAT_FIELDS = ('total_cost', 'total_regret', 'row_oscillation', 'realization_bound', 'runtime_seconds', 'solve_seconds')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def expected_files():
    files = {f'{key}.{ext}' for key in DATASETS for ext in ('txt', 'data.gz', 'labels0.gz')}
    files |= {f'{key.split("/")[0]}/README.txt' for key in DATASETS}
    return sorted(files | {'tag_reference.json', 'mnist/original_t10k_labels.gz'})


@lru_cache(maxsize=6)
def load_dataset(root, dataset):
    root = Path(root)
    x = np.loadtxt(root / (dataset + '.data.gz'))
    labels = np.loadtxt(root / (dataset + '.labels0.gz'), dtype=np.int64)
    if x.shape != DATASETS[dataset] or labels.shape != (len(x),) or not np.isfinite(x).all():
        raise RuntimeError(f'STOP: invalid benchmark dimensions/values for {dataset}: {x.shape}')
    return x, labels


def validate_sources(root):
    root = Path(root)
    files = {name: {'sha256': sha256(root / name), 'bytes': (root / name).stat().st_size,
                    'url': ('https://raw.githubusercontent.com/gagolews/clustering-data-v1/' + COMMIT + '/' + name)
                    if name not in ('tag_reference.json', 'mnist/original_t10k_labels.gz') else
                    ('https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz' if name.startswith('mnist/') else
                     'https://api.github.com/repos/gagolews/clustering-data-v1/git/ref/tags/v1.2.0')}
             for name in expected_files()}
    if json.loads((root / 'tag_reference.json').read_text())['object']['sha'] != COMMIT:
        raise RuntimeError('STOP: benchmark tag changed')
    inventory = {}
    for name in DATASETS:
        x, y = load_dataset(str(root), name)
        ids, counts = np.unique(y, return_counts=True)
        inventory[name] = dict(shape=list(x.shape), stored_label_counts=dict(zip(map(str, ids), map(int, counts))),
                               noise_count=int(np.sum(y == 0)), noise_proportion=float(np.mean(y == 0)))
    raw = gzip.decompress((root / 'mnist/original_t10k_labels.gz').read_bytes())
    magic, count = struct.unpack('>II', raw[:8])
    original = np.frombuffer(raw, dtype=np.uint8, offset=8)
    _, labels = load_dataset(str(root), 'mnist/digits10k')
    expected_labels = np.where(original == 0, 10, original)
    if magic != 2049 or count != 10000 or not np.array_equal(labels, expected_labels):
        raise RuntimeError('STOP: MNIST semantic-label mapping could not be verified')
    for digit in (6, 9):
        if np.sum(labels == digit) < 512:
            raise RuntimeError(f'STOP: fewer than 512 distinct digit {digit} records')
    return dict(version='v1.2.0', commit=COMMIT, files=files, datasets=inventory,
                mnist_semantic_mapping={str(i): i % 10 for i in range(1, 11)},
                mnist_mapping_validation='all 10000 labels match original test-label IDX in original row order; digit zero is stored as 10',
                source_row_index='zero based original file line number')


def settings():
    result = []
    for dataset in DATASETS:
        result.append(dict(setting_id='aggregate_' + dataset.replace('/', '_'), dataset=dataset,
                           n=64, d=DATASETS[dataset][1], condition='mixed', feature_order='native',
                           memberships=[dict(panel='aggregate', condition=dataset)]))
    def focused(sid, n, d, condition, memberships):
        result.append(dict(setting_id=sid, dataset='mnist/digits10k', n=n, d=d,
                           condition=condition, feature_order='shared_pixel_permutation', memberships=memberships))
    for condition in ('6-to-6', '9-to-9', '6-to-9', '9-to-6'):
        memberships = [dict(panel='label', condition=condition)]
        if condition == '6-to-6':
            memberships.append(dict(panel='sample_size', condition=condition))
        focused('label_' + condition, 64, 784, condition, memberships)
    for d in DIMENSIONS:
        memberships = [dict(panel='dimension', condition='6-to-6')]
        if d == 784:
            memberships.append(dict(panel='sample_size', condition='6-to-6'))
        focused(f'dimension_d{d}', 256, d, '6-to-6', memberships)
    for n in (16, 32, 128):
        focused(f'sample_size_n{n}', n, 784, '6-to-6', [dict(panel='sample_size', condition='6-to-6')])
    return result


def design(pilot=False):
    return dict(schema=NAME, phase='pilot' if pilot else 'production',
                parents=1 if pilot else 20, partitions=1 if pilot else 5, K=3 if pilot else 50,
                epsilon=[.25, 4., 16.] if pilot else [.25, .5, 1., 2., 4., 8., 16.],
                settings=settings(), mechanisms={str(p): list(arms) for p, arms in ARMS.items()},
                seed_namespace=SEED_NAMESPACE, feature_seed=str(seed(SEED_NAMESPACE, 'public-feature-order')),
                normalization=NORMALIZATION, objective='sum',
                gap_zero_rule='abs(gap) <= 64*float64_eps*max(1,abs(OPT),abs(RAND))',
                small_gap_rule='0 < gap <= 1e-6*max(1,abs(OPT),abs(RAND))',
                privacy_metric='sum_record_euclidean_in_standardized_coordinates',
                privacy_scope='standardized inputs only; no raw-data preprocessing privacy guarantee',
                noise_protocol='independent arm channels; explicit unit-noise coupling across epsilon and supported objectives; endpoint reuse recorded',
                tie_rule='fixed scipy linear_sum_assignment input order',
                seed_algorithm='SHA256 of JSON key tuple; first 16 bytes interpreted little-endian; NumPy default_rng PCG64',
                inference='noise then partition then parent mean; pointwise Student-t 95% intervals over parent means',
                association_epsilon=4., association_objectives=[1, 2],
                geometry_range_edges=dict(rank=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                                          rank_ratio=[0, .125, .25, .5, .75, 1.000000000001],
                                          centered_rms=[0, .125, .25, .5, .75, 1., 1.5, 2.],
                                          side_center_separation=[0, .125, .25, .5, 1., 2., 4.]),
                geometry_range_interpretation='exploratory observed mean comparisons within fixed ranges; separate panels, label conditions, epsilon and objective; no universal or causal claim')


def all_units(cfg):
    return [dict(s, parent=parent, partition=part,
                 instance_id=f'{s["setting_id"]}_parent{parent:02d}_part{part}')
            for s in cfg['settings'] for parent in range(cfg['parents']) for part in range(cfg['partitions'])]


def environment():
    versions = {key: importlib.metadata.version(key) for key in ('numpy', 'scipy', 'pandas', 'matplotlib', 'pytest')}
    versions.update(python=platform.python_version(), machine=platform.machine(),
                    system=platform.system(), libc=list(platform.libc_ver()), numerical_threads=1)
    versions['numpy_build_configuration'] = np.__config__.CONFIG
    versions['numerical_binary_sha256'] = {
        name: sha256(importlib.util.find_spec(name).origin)
        for name in ('numpy._core._multiarray_umath', 'numpy.linalg._umath_linalg', 'scipy.optimize._lsap')}
    return versions


def resolve(root, source_data, pilot=False):
    cfg = design(pilot)
    sources = validate_sources(source_data)
    package = Path(__file__).parent
    hashes = {str(p.relative_to(package.parent)): sha256(p) for p in sorted(package.glob('*.py'))}
    payload = dict(kind=NAME, design=cfg, source_sha256=hashes, versions=environment(), sources=sources)
    fp = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    payload.update(fingerprint=fp, source_data=str(Path(source_data).resolve()))
    return Path(root) / NAME / cfg['phase'] / fp, payload


def pixel_order():
    return np.random.default_rng(seed(SEED_NAMESPACE, 'public-feature-order')).permutation(784)


def parent_lists(cfg, dataset, parent, partition):
    _, labels = load_dataset(cfg['source_data'], dataset)
    key = (SEED_NAMESPACE, cfg['phase'], dataset, parent)
    if cfg.get('focused', False):
        lists, pools = {}, {}
        for digit in (6, 9):
            available = np.flatnonzero(labels == digit)
            pool = np.random.default_rng(seed(*key, 'parent', digit)).choice(available, 512, replace=False)
            order = np.random.default_rng(seed(*key, 'partition', partition, digit)).permutation(pool)
            lists[f'Q{digit}'], lists[f'R{digit}'] = order[:256], order[256:]
            pools[str(digit)] = pool
        return lists, pools
    pool = np.random.default_rng(seed(*key, 'aggregate-parent')).choice(len(labels), 128, replace=False)
    order = np.random.default_rng(seed(*key, 'aggregate-partition', partition)).permutation(pool)
    return {'Q': order[:64], 'R': order[64:]}, {'mixed': pool}


def source_instance(manifest, unit):
    cfg = dict(manifest['design'], source_data=manifest['source_data'], focused=unit['condition'] != 'mixed')
    lists, pools = parent_lists(cfg, unit['dataset'], unit['parent'], unit['partition'])
    n = unit['n']
    if unit['condition'] == 'mixed':
        qi, ri = lists['Q'][:n], lists['R'][:n]
    else:
        qdigit, rdigit = unit['condition'].split('-to-')
        qi, ri = lists['Q' + qdigit][:n], lists['R' + rdigit][:n]
    if len(qi) != n or len(ri) != n or len(np.unique(np.r_[qi, ri])) != 2*n:
        raise RuntimeError('STOP: nonunique or overlapping source rows')
    features = np.arange(unit['d']) if unit['feature_order'] == 'native' else pixel_order()[:unit['d']]
    data, labels = load_dataset(manifest['source_data'], unit['dataset'])
    Q, R = data[qi][:, features], data[ri][:, features]
    records = dict(q_source_ids=qi, r_source_ids=ri, feature_ids=features, q_stored_labels=labels[qi],
                   r_stored_labels=labels[ri], **{'maximum_' + key: value for key, value in lists.items()},
                   **{'parent_' + key: value for key, value in pools.items()})
    if unit['dataset'] == 'mnist/digits10k':
        records.update(q_semantic_labels=labels[qi] % 10, r_semantic_labels=labels[ri] % 10)
    return Q, R, records


def standardize(Q, R):
    pooled = np.concatenate((Q, R), axis=0)
    center = pooled.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum((pooled-center)**2, axis=1))))
    if not np.isfinite(scale):
        raise RuntimeError('STOP: nonfinite pooled radius')
    if scale == 0:
        return None, None, center, scale
    q, r = (Q-center)/scale, (R-center)/scale
    joined = np.r_[q, r]
    if not np.allclose(joined.mean(axis=0), 0, atol=2e-12) or not np.isclose(np.mean(np.sum(joined**2, axis=1)), 1., atol=2e-12):
        raise RuntimeError('STOP: joint normalization invariant failed')
    return q, r, center, scale


def spectrum(x):
    n, d = x.shape
    centered = x - x.mean(axis=0)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    threshold = float(singular[0]*max(n, d)*np.finfo(float).eps) if len(singular) else 0.
    rank = int(np.sum(singular > threshold))
    # Participation ratio uses ALL covariance eigenvalues, without rank truncation.
    eigenvalues = singular**2 / (n-1) if n > 1 else np.zeros_like(singular)
    trace = float(eigenvalues.sum())
    effective = float(trace**2 / (eigenvalues @ eigenvalues)) if trace > 0 else None
    if rank > min(d, n-1):
        raise RuntimeError('STOP: covariance rank ceiling failed')
    return dict(n=n, d=d, covariance_rank=rank, effective_rank=effective,
                effective_rank_over_rank=effective/rank if effective is not None and rank else None,
                status='ok' if rank else 'zero_variance', covariance_trace=trace,
                covariance_eigenvalues=eigenvalues.tolist(), singular_values=singular.tolist(),
                singular_value_rank_threshold=threshold,
                rank_rule='singular > largest_singular*max(n,d)*float64_eps',
                effective_rank_spectrum='all centered sample covariance eigenvalues, denominator n-1, no truncation',
                centered_rms_radius=float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))),
                mean_relative_to_pooled_center=x.mean(axis=0).tolist())


def diagnostics(x, labels, mixed):
    result = spectrum(x)
    ids, counts = np.unique(labels, return_counts=True)
    result.update(label_counts=dict(zip(map(str, ids), map(int, counts))), noise_count=int(np.sum(labels == 0)),
                  noise_proportion=float(np.mean(labels == 0)))
    result['within_label'] = {str(label): spectrum(x[labels == label]) for label in ids if mixed and label != 0}
    return result


def gap_info(optimum, random):
    gap = float(random-optimum)
    tolerance = float(64*np.finfo(float).eps*max(1., abs(optimum), abs(random)))
    if gap < -tolerance:
        raise RuntimeError('STOP: materially negative RAND minus OPT')
    status = 'zero_gap' if abs(gap) <= tolerance else ('small_positive_gap' if gap <= 1e-6*max(1., abs(optimum), abs(random)) else 'ok')
    return dict(value=gap, tolerance=tolerance, status=status)


def checkpoint(directory, unit, fingerprint, verify=False):
    marker = directory / 'units' / (unit['instance_id'] + '.json')
    if not marker.exists():
        return None
    meta = json.loads(marker.read_text())
    if meta['fingerprint'] != fingerprint or meta['unit'] != unit:
        raise RuntimeError('checkpoint identity mismatch: ' + str(marker))
    for name, rec in meta['files'].items():
        path = directory / 'units' / name
        if not path.exists() or path.stat().st_size != rec['bytes'] or (verify and sha256(path) != rec['sha256']):
            raise RuntimeError('committed checkpoint corrupt: ' + str(path))
    return meta


def run_unit(directory, manifest, unit, *, prepared_instance=None, seed_namespace=None, extra_metadata=None, reused_draw=None):
    refuse_login_compute('run a clustering benchmark unit')
    if checkpoint(directory, unit, manifest['fingerprint'], verify=True):
        print('REUSE', unit['instance_id'], flush=True)
        return
    started = time.perf_counter()
    cfg = manifest['design']; n, d = unit['n'], unit['d']
    Q, R, records = source_instance(manifest, unit) if prepared_instance is None else prepared_instance
    q, r, center, scale = standardize(Q, R)
    stem = directory / 'units' / unit['instance_id']
    meta = dict(fingerprint=manifest['fingerprint'], unit=unit, normalization=NORMALIZATION,
                source_radius=scale, pooled_center=center.tolist(),
                host=socket.gethostname(), job_id=os.environ.get('SLURM_JOB_ID'),
                array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'), optima={}, random_expected={}, gaps={},
                diagnostics={}, public={}, bounds={}, reuse={}, files={}, degenerate=scale == 0,
                source_data=manifest['source_data'])
    if extra_metadata:
        meta.update(extra_metadata)
    if scale == 0:
        meta.update(status='degenerate_zero_pooled_radius', zero_assignment_gap=True, actual_solves=0, raw_rows=0)
        save_npz(str(stem) + '.geometry.npz', raw_Q=Q, raw_R=R, pooled_center=center, **records)
    else:
        geo = Geometry.build(r, q)
        meta['public'] = geo.public_metadata()
        meta['cross_distance_summary'] = dict(zip(('minimum', 'q10', 'median', 'q90', 'maximum'),
                                                 map(float, np.quantile(geo.C1, [0., .1, .5, .9, 1.]))))
        meta['cross_distance_summary'].update(mean=float(geo.C1.mean()), sd=float(geo.C1.std()),
                                             side_center_separation=float(np.linalg.norm(q.mean(axis=0)-r.mean(axis=0))))
        meta['diagnostics'] = {role: diagnostics(x, records[role.lower()+'_stored_labels'], unit['condition'] == 'mixed')
                               for role, x in (('Q', q), ('R', r))}
        if meta['diagnostics']['Q']['covariance_rank'] != geo.rank:
            raise RuntimeError('STOP: diagnostic and mechanism Q ranks disagree')
        # Scalar normalization and translation preserve rank and effective rank.
        for role, raw in (('Q', Q), ('R', R)):
            original = spectrum(raw)
            standardized = meta['diagnostics'][role]
            if original['covariance_rank'] != standardized['covariance_rank']:
                raise RuntimeError('STOP: normalization changed numerical rank; needs review')
            if original['effective_rank'] is not None and not np.isclose(original['effective_rank'], standardized['effective_rank'], rtol=1e-10, atol=1e-10):
                raise RuntimeError('STOP: normalization changed effective rank')
        shape = (2, 7, len(cfg['epsilon']), cfg['K'])
        arrays = {field: np.empty(shape) for field in FLOAT_FIELDS}
        arrays['permutations'] = np.empty((*shape, n), dtype=np.uint16)
        arrays['optimum_permutations'] = np.stack([solve(geo.C1), solve(geo.C2)]).astype(np.uint16)
        optima = {p: float(C[np.arange(n), arrays['optimum_permutations'][ip]].sum())
                  for ip, (p, C) in enumerate(((1, geo.C1), (2, geo.C2)))}
        meta['optima'] = {str(p): v for p, v in optima.items()}
        meta['random_expected'] = {str(p): float(C.sum()/n) for p, C in ((1, geo.C1), (2, geo.C2))}
        meta['gaps'] = {str(p): gap_info(optima[p], meta['random_expected'][str(p)]) for p in (1, 2)}
        meta['preparation_seconds'] = time.perf_counter()-started
        for draw in range(cfg['K']):
            def rng_for(channel):
                return np.random.default_rng(seed(seed_namespace or SEED_NAMESPACE, cfg['phase'], unit['instance_id'], 'unit', draw, channel))
            channels, _ = unit_channels(geo, rng_for)
            for ie, epsilon in enumerate(cfg['epsilon']):
                # A benchmark is not the synthetic U cube: no synthetic squared cap.
                precomputed = reused_draw(draw, ie, epsilon, geo, optima) if reused_draw else None
                results = evaluate_draw(geo, channels, epsilon, 'benchmark', optima, precomputed)
                for ip, p in enumerate((1, 2)):
                    for ia, arm in enumerate(ARMS[p]):
                        item = results[p, arm]
                        for field in FLOAT_FIELDS:
                            arrays[field][ip, ia, ie, draw] = item[field]
                        pi = item['permutation']
                        arrays['permutations'][ip, ia, ie, draw] = pi
                        C = geo.C1 if p == 1 else geo.C2
                        raw_regret = float(C[np.arange(n), pi].sum()-optima[p])
                        if raw_regret < -64*np.finfo(float).eps*max(1., optima[p], item['total_cost']):
                            raise RuntimeError('STOP: materially negative realized regret')
                        # Preserve the actual computed difference, including harmless roundoff.
                        arrays['total_regret'][ip, ia, ie, draw] = raw_regret
                        if draw == 0:
                            meta['reuse'][f'{p}:{arm}'] = {key: item[key] for key in ('source_arm', 'reuse_reason')}
                            cert = bound(geo, arm, p, epsilon, 'benchmark')
                            meta['bounds'].setdefault(f'{p}:{arm}', []).append({key: (None if isinstance(value, float) and not np.isfinite(value) else value) for key, value in cert.items()})
        if not np.all(np.sort(arrays['permutations'], axis=-1) == np.arange(n)):
            raise RuntimeError('STOP: invalid bijection')
        for ip, C in enumerate((geo.C1, geo.C2)):
            rebuilt = C[np.arange(n), arrays['permutations'][ip]].sum(axis=-1)
            if not np.allclose(rebuilt, arrays['total_cost'][ip], rtol=1e-12, atol=1e-10):
                raise RuntimeError('STOP: saved assignment cost mismatch')
        save_npz(str(stem) + '.geometry.npz', Q=q, R=r, pooled_center=center, source_radius=np.array(scale),
                 public_B=geo.B, public_singular=geo.singular, **records)
        save_npz(str(stem) + '.npz', **arrays)
        meta['actual_solves'] = 2 + cfg['K']*len(cfg['epsilon'])*sum(not rec['source_arm'] for rec in meta['reuse'].values())
        meta['raw_rows'] = 2 + 2*7*len(cfg['epsilon'])*cfg['K']
        meta['status'] = 'complete'
    for path in stem.parent.glob(stem.name + '.*npz'):
        meta['files'][path.name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    meta['elapsed_seconds'] = time.perf_counter()-started
    meta['process_peak_rss_kib'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    atomic_json(str(stem) + '.json', meta)
    print(f'COMPLETE {unit["instance_id"]} {meta["elapsed_seconds"]:.3f}s', flush=True)


def prepare(root, source_data, pilot):
    directory, manifest = resolve(root, source_data, pilot)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'units').mkdir(exist_ok=True)
    marker = directory / 'manifest.json'
    if marker.exists() and json.loads(marker.read_text()) != manifest:
        raise RuntimeError('manifest conflict')
    atomic_json(marker, manifest)
    atomic_json(directory / 'expected_units.json', all_units(manifest['design']))
    np.save(directory / 'feature_order.npy', pixel_order(), allow_pickle=False)
    # uv-managed environments need not contain pip. Inspect installed metadata
    # without installing or changing the scientific environment.
    distributions = sorted(importlib.metadata.distributions(), key=lambda d: d.metadata['Name'].lower())
    (directory / 'requirements.freeze.txt').write_text(''.join(f'{d.metadata["Name"]}=={d.version}\n' for d in distributions))
    atomic_json(directory / 'installed_distribution_origins.json',
                {d.metadata['Name']: json.loads(d.read_text('direct_url.json'))
                 for d in distributions if d.read_text('direct_url.json')})
    (directory / 'numpy_build_configuration.txt').write_text(str(np.__config__.CONFIG))
    print('Resolved clustering benchmark results:', directory, flush=True)
    return directory, manifest


def run_study(root, source_data, pilot=False, stage='all', shard_index=None, shard_count=None, directory=None):
    refuse_login_compute('run the clustering benchmark')
    if stage in ('prepare', 'all'):
        directory, manifest = prepare(root, source_data, pilot)
    elif directory:
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
    else:
        directory, manifest = resolve(root, source_data, pilot)
        if not (directory / 'manifest.json').exists():
            raise RuntimeError('prepare required')
    print('Resolved clustering benchmark results:', directory, flush=True)
    if stage in ('shard', 'all'):
        index, count = (0 if shard_index is None else shard_index), (1 if shard_count is None else shard_count)
        if not 0 <= index < count:
            raise ValueError('invalid shard')
        # Refuse source or environment drift relative to the frozen run.
        _, current = resolve(root, manifest['source_data'], manifest['design']['phase'] == 'pilot')
        if current['fingerprint'] != manifest['fingerprint']:
            raise RuntimeError('STOP: source/environment/data fingerprint drift')
        for unit in all_units(manifest['design'])[index::count]:
            run_unit(directory, manifest, unit)
    if stage in ('finalize', 'all'):
        from .clustering_benchmark_report import finalize
        finalize(directory)
    return directory
