"""Portable scheduling around unchanged production mechanism implementations.

Preparation uses exact archived sampling/transform artifacts. This adapter omits
historical comparisons to experiments outside the paper and their absolute paths.
It never reads saved production outcomes to compute a rerun outcome.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import copy
import hashlib
import json
import multiprocessing
import shutil

from common import ROOT, DATA, STUDIES, configure, digest, environment, frozen, read_json, write_json

DATASETS = ('graves/ring_noisy', 'wut/circles', 'fcps/chainlink', 'other/iris',
    'uci/banknote', 'uci/ecoli', 'uci/yeast', 'uci/glass', 'uci/wine', 'uci/statlog',
    'uci/wdbc', 'uci/ionosphere', 'uci/sonar', 'sipu/worms_64', 'mnist/digits10k', 'mnist/fashion10k')

MODULE = MANIFEST = RUN = CONTEXT = None


def portable_manifest(study, regenerate_null=False, experiment=None, dataset='all', mode='paper'):
    original = read_json(ROOT/'provenance'/study/'manifest.json')
    result = copy.deepcopy(original)
    result['production_fingerprint'] = result.pop('fingerprint')
    result['adapter'] = 'portable-paper-only-v1'
    result['adapter_options'] = dict(regenerate_null=bool(regenerate_null and study == 'sonar'))
    cfg = result['design']
    if study == 'euclidean':
        cfg['settings'] = [dict(s, objectives=[1]) for s in cfg['settings']
                           if s['panel'] in ('epsilon', 'sample_size')]
    if study == 'fashion':
        # This key enabled extra epsilon-8 validation, not the epsilon-16 law.
        cfg.pop('geometry_reference', None)
    if study == 'sonar':
        cfg['epsilon'] = [1., 2., 4., 8., 16.]
        cfg['arms'] = ['gumbel', 'ambient_radial', 'span_radial', 'cost_infinity']
        cfg['B'] = 1
    result['public_release'] = dict(experiment=experiment, dataset=dataset, mode=mode,
        quick_is_production_subset=True, seeds_independent_of_workers=True)
    if study in ('euclidean', 'squared'):
        panel = 'sample_size' if experiment == 'size-sweep' else 'epsilon'
        cfg['settings'] = [s for s in cfg['settings'] if s['panel'] == panel
                           and (dataset == 'all' or s['dataset'] == dataset)]
        if not cfg['settings']:
            raise ValueError('No settings match this experiment and dataset')
    if mode == 'quick':
        cfg.update(K=10 if study == 'sonar' else 3)
        if study != 'sonar':
            cfg.update(parents=2, partitions=1)
        if study == 'fashion':
            cfg.update(chains=1, dimensions=[104, 24, 4])
        if study in ('euclidean', 'squared'):
            if experiment == 'size-sweep':
                cfg['settings'] = [s for s in cfg['settings'] if s['n'] in (20, 60, 200)]
            else:
                cfg['settings'] = [dict(s, epsilon=[1., 8., 16.]) for s in cfg['settings']]
    result['fingerprint'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:24]
    return result


def tasks(study, manifest):
    cfg = manifest['design']
    if study in ('euclidean', 'squared'):
        return [('unit', i, p, q) for i in range(len(cfg['settings']))
                for p in range(cfg['parents']) for q in range(cfg['partitions'])]
    if study == 'fashion':
        return [('fashion', p, c, d, q) for p in range(cfg['parents'])
                for c in range(cfg['chains']) for d in cfg['dimensions'] for q in range(cfg['partitions'])]
    return [('sonar', k) for k in range(cfg['K'])]


def initialize(study, run):
    global MODULE, MANIFEST, RUN, CONTEXT
    configure(study)
    MODULE = frozen(study)
    RUN = Path(run)
    MANIFEST = read_json(RUN/'manifest.json')
    if study == 'sonar':
        import numpy as np
        from private_matching.integral_mechanisms import Geometry
        with np.load(RUN/'context.npz') as z:
            CONTEXT = {key: z[key] for key in z.files}
        CONTEXT['geo'] = Geometry.build(CONTEXT['R'], CONTEXT['Q'])


def execute(task):
    import numpy as np
    kind, *items = task
    if kind == 'basis':
        parent, chain = items
        meta = MODULE.make_basis(RUN, MANIFEST, parent, chain, regenerate=True)
        expected = read_json(ROOT/'inputs/fashion/basis_checks.json')[f'p{parent:02d}_c{chain}']
        # Array hashes avoid ZIP metadata differences between NumPy releases.
        with np.load(RUN/'bases'/f'p{parent:02d}_c{chain}.npz') as z:
            actual = {key: hashlib.sha256(np.ascontiguousarray(z[key]).view(np.uint8)).hexdigest()
                      for key in ('O', 'W', 'radii')}
        names = ('O', 'W', 'radii') if MANIFEST['public_release']['mode'] == 'paper' else ('O', 'W')
        return dict(parent=parent, chain=chain,
                    matches_archived_arrays=all(actual[k] == expected[k] for k in names), hashes=actual)
    if kind == 'unit':
        setting, parent, partition = items
        spec = MANIFEST['design']['settings'][setting]
        group = dict(setting=spec, parent=parent, group_id=spec['setting_id']+f'_parent{parent:02d}')
        MODULE.run_unit(RUN, MANIFEST, group, partition)
        return None
    if kind == 'fashion':
        MODULE.run_instance(RUN, MANIFEST, *items)
        return None
    draw = items[0]
    output = RUN/'units'/f'draw{draw:04d}.csv'
    if output.exists() and output.with_suffix('.json').exists():
        # Checkpoint CSV is published only after the corresponding arrays.
        check = read_json(output.with_suffix('.json'))
        if (check.get('fingerprint') != MANIFEST['fingerprint'] or digest(output) != check['csv_sha256']
                or digest(output.with_suffix('.npz')) != check['arrays_sha256']):
            raise RuntimeError(f'Changed Sonar checkpoint: {output}')
        return None
    data, cfg, sr = CONTEXT, MANIFEST['design'], MODULE
    geo = data['geo']
    noise = sr.channels(geo, cfg['phase'], draw)
    rows, permutations = [], []
    for arm in cfg['arms']:
        for epsilon in cfg['epsilon']:
            scores = sr.squared_scores(geo, noise, arm, epsilon)
            pi = sr.prioritized_solve(scores,
                sr.rng(cfg['phase'], 'private', draw, arm, epsilon, 'assignment-priorities'))
            if not np.array_equal(np.sort(pi), np.arange(cfg['N'])):
                raise RuntimeError('Incomplete Sonar matching')
            T = float(sr.energy_statistics(data['grid_distances'], sr.slot_membership(pi, data['metal']),
                         cfg['a'], cfg['b'])[0])
            p = float(sr.bank_pvalues(T, data['null_sorted'][None, :])[0])
            rows.append(dict(mechanism=arm, epsilon=epsilon, draw=draw, rank_T=T, primary_p=p))
            permutations.append(pi)
    import pandas as pd
    np.savez_compressed(output.with_suffix('.npz'), permutations=np.asarray(permutations, dtype=np.uint16))
    temporary = output.with_suffix('.csv.tmp')
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(output)
    write_json(output.with_suffix('.json'), dict(csv_sha256=digest(output),
               arrays_sha256=digest(output.with_suffix('.npz')), fingerprint=MANIFEST['fingerprint'],
               draw=draw, hardware=sr.hardware()))
    return None


def dispatch(study, run, work, jobs):
    if jobs == 1:
        initialize(study, run)
        return [execute(task) for task in work]
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(study, str(run))) as pool:
        return list(pool.map(execute, work, chunksize=1))


def prepare(study, run, manifest, jobs, regenerate_null):
    import numpy as np
    run.mkdir(parents=True, exist_ok=True)
    if (run/'manifest.json').exists() and read_json(run/'manifest.json') != manifest:
        raise RuntimeError('This output directory belongs to another configuration; use a fresh directory.')
    for name in ('masters', 'units', 'bases', 'aggregates'):
        (run/name).mkdir(exist_ok=True)
    write_json(run/'manifest.json', manifest)
    write_json(run/'execution_environment.json', environment())
    if study in ('euclidean', 'squared'):
        registry = read_json(ROOT/'inputs'/study/'master_registry.json')
        source = ROOT/'inputs'/('euclidean' if study == 'squared' else study)
        module = frozen(study)
        base = module.base if study == 'squared' else module
        needed = {base.master_key(s, p) for s in manifest['design']['settings']
                  for p in range(manifest['design']['parents'])}
        registry = {k: v for k, v in registry.items() if k in needed}
        for record in registry.values():
            original = source/record['file']
            if digest(original) != record['sha256']:
                raise RuntimeError(f'Changed archived master: {original}')
            target = run/record['file']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        write_json(run/'master_registry.json', registry)
    if study == 'fashion':
        from input_artifacts import materialize_fashion
        materialize_fashion(run, parents=range(manifest['design']['parents']))
        cfg = manifest['design']
        checks = dispatch(study, run, [('basis', p, c) for p in range(cfg['parents'])
                            for c in range(cfg['chains'])], jobs)
        write_json(run/'geometry_comparison.json', dict(all_archived_array_hashes_match=
                   all(c['matches_archived_arrays'] for c in checks), checks=checks))
    if study == 'sonar':
        sr = frozen(study)
        from private_matching.integral_mechanisms import Geometry
        from scipy.spatial.distance import cdist
        R, metal, stored, sources = sr.validate_sources(DATA/'sonar')
        Q = sr.public_grid()
        geo = Geometry.build(R, Q)
        pi = sr.prioritized_solve(geo.C2, sr.rng('nonprivate', 'assignment-priorities'))
        cfg = manifest['design']
        distances = cdist(Q, Q)
        T = float(sr.energy_statistics(distances, sr.slot_membership(pi, metal), cfg['a'], cfg['b'])[0])
        if regenerate_null:
            allocations = sr.label_allocations(cfg['J'], cfg['N'], cfg['a'], sr.rng(cfg['phase'], 'public-bank', 0))
            null = sr.energy_statistics(distances, allocations, cfg['a'], cfg['b'])
        else:
            with np.load(ROOT/'inputs/sonar/primary_null.npz') as z:
                null = z['statistics']
        ordered = np.sort(null)
        p = float(sr.bank_pvalues(T, ordered[None, :])[0])
        np.savez_compressed(run/'context.npz', R=R, Q=Q, metal=metal, grid_distances=distances,
                            null_sorted=ordered, nonprivate_T=T, nonprivate_p=p, public_rank=geo.rank)
        write_json(run/'null_provenance.json', dict(regenerated_from_seed=regenerate_null,
                   primary_bank=1, J=cfg['J'], original_seed_key=['production', 'public-bank', 0]))


def aggregate(study, run, manifest):
    import numpy as np
    import pandas as pd
    cfg = manifest['design']
    if study == 'sonar':
        frames = []
        for draw in range(cfg['K']):
            path = run/'units'/f'draw{draw:04d}.csv'
            check = read_json(path.with_suffix('.json'))
            if digest(path) != check['csv_sha256']:
                raise RuntimeError(f'Changed checkpoint: {path}')
            frames.append(pd.read_csv(path, float_precision='round_trip'))
        frame = pd.concat(frames, ignore_index=True)
        frame.to_csv(run/'aggregates/draws.csv.gz', index=False, compression={'method':'gzip', 'mtime':0})
        with np.load(run/'context.npz') as z:
            write_json(run/'aggregates/baseline.json', {k: float(z[k]) for k in ('nonprivate_T','nonprivate_p','public_rank')})
        return
    module = frozen(study)
    rows = []
    if study == 'fashion':
        for _, parent, chain, d, part in tasks(study, manifest):
            stem = run/'units'/f'p{parent:02d}_c{chain}_d{d:03d}_part{part}'
            meta = read_json(stem.with_suffix('.json'))
            if digest(stem.with_suffix('.npz')) != meta['sha256']:
                raise RuntimeError(f'Changed result: {stem}')
            with np.load(stem.with_suffix('.npz')) as z:
                for ia, arm in enumerate(module.ARMS):
                    regret = float(z['regret'][ia].mean())
                    opt, rand = float(z['optimum']), float(z['RAND'])
                    rows.append(dict(parent=parent, chain=chain, d=d, partition=part, mechanism=arm,
                        regret=regret, regret_per_pair=regret/cfg['n'], gap_fraction=regret/(rand-opt),
                        OPT=opt, RAND_gap=rand-opt, public_rank=meta['public_rank'], effective_rank=meta['effective_rank'],
                        projected_RMS=meta['projected_RMS'], retained_energy=meta['retained_energy'],
                        total_operator_norm=meta['total_operator_norm'], input_RMS=meta['input_RMS']))
        pd.DataFrame(rows).to_csv(run/'aggregates/instance_summary.csv.gz', index=False,
                                 compression={'method':'gzip', 'mtime':0})
        return
    base = module.base if study == 'squared' else module
    for _, index, parent, part in tasks(study, manifest):
        setting = cfg['settings'][index]
        instance = setting['setting_id']+f'_parent{parent:02d}_part{part}'
        meta = read_json(run/'units'/(instance+'.json'))
        for name, checksum in meta['files'].items():
            if digest(run/'units'/name) != checksum:
                raise RuntimeError(f'Changed result: {name}')
        with np.load(run/'units'/(instance+'.npz')) as z:
            for ia, arm in enumerate(base.ARMS):
                for ie, epsilon in enumerate(setting['epsilon']):
                    opt, rand = float(z['optimum'][0]), float(z['random_expected'][0])
                    rows.append(dict(setting_id=setting['setting_id'], panel=setting['panel'], dataset=setting['dataset'],
                        n=setting['n'], d=setting['d'], parent=parent, partition=part, objective_p=setting['objectives'][0],
                        mechanism=arm, epsilon=epsilon, K=cfg['K'],
                        mean_total_cost=float(z['total_cost'][0,ia,ie].mean()),
                        mean_total_regret=float(z['total_regret'][0,ia,ie].mean()), optimum=opt,
                        random_expected=rand, random_minus_optimum=rand-opt, expected_bound=float(z['expected_bound'][0,ia,ie])))
    frame = pd.DataFrame(rows)
    keys = ['setting_id','panel','dataset','n','d','parent','objective_p','mechanism','epsilon']
    metrics = ['mean_total_cost','mean_total_regret','optimum','random_expected','random_minus_optimum','expected_bound']
    if not (frame.groupby(keys).size() == cfg['partitions']).all():
        raise RuntimeError('Missing or duplicated partition results')
    parent = frame.groupby(keys, sort=True)[metrics].mean().reset_index()
    parent['partition_count'] = cfg['partitions']
    parent.to_csv(run/'aggregates/parent_summaries.csv.gz', index=False, compression={'method':'gzip','mtime':0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=STUDIES)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--experiment', choices=('benchmarks', 'size-sweep', 'dimension-sweep', 'sonar'))
    parser.add_argument('--dataset', choices=('all', *DATASETS), default='all')
    parser.add_argument('--mode', choices=('quick', 'paper'), default='paper')
    parser.add_argument('--list-datasets', action='store_true')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--stage', choices=('all','prepare','run','aggregate'), default='all')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--regenerate-null', action='store_true')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if args.list_datasets:
        print('\n'.join(DATASETS))
        return
    if args.study is None or args.output is None:
        parser.error('--study and --output are required')
    manifest = portable_manifest(args.study, args.regenerate_null, args.experiment, args.dataset, args.mode)
    work = tasks(args.study, manifest)[args.shard_index::args.shard_count]
    if args.plan:
        cfg = manifest['design']
        print(json.dumps(dict(study=args.study, tasks=len(work), original_fingerprint=manifest['production_fingerprint'],
              mode=args.mode, stage=args.stage, workers=args.jobs, draws_per_instance=cfg['K'],
              output=str(args.output), design=cfg,
              projection_bases=cfg['parents']*cfg['chains'] if args.study=='fashion' else 0,
              experiments_executed=False), indent=2))
        return
    if args.shard_count > 1 and args.stage == 'all':
        parser.error('For sharding use prepare once, then run per shard, then aggregate once.')
    configure(args.study)
    if args.stage in ('prepare','all'):
        if args.study in ('fashion', 'sonar'):
            from fetch_data import fetch
            fetch('mnist/fashion10k' if args.study == 'fashion' else 'uci/sonar')
        prepare(args.study, args.output, manifest, args.jobs, args.regenerate_null)
    if not (args.output/'manifest.json').exists() or read_json(args.output/'manifest.json') != manifest:
        raise RuntimeError('Prepare this run before launching workers or aggregation.')
    if args.stage in ('run','all'):
        dispatch(args.study, args.output, work, args.jobs)
    if args.stage in ('aggregate','all'):
        aggregate(args.study, args.output, manifest)


if __name__ == '__main__':
    main()
