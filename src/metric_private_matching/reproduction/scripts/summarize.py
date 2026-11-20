"""Export tidy results from saved production data or a completed independent run."""
from pathlib import Path
import argparse
import json

from common import ROOT, configure, read_json, write_json


def interval(values):
    import numpy as np
    from scipy.stats import t
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.
    return mean, mean-half, mean+half


def summarize(study, source, *, experiment, saved=False):
    import pandas as pd
    import numpy as np
    run = None if saved else read_json(source.parent/'manifest.json')
    cfg = None if saved else run['design']
    common = dict(experiment=experiment, mode='paper' if saved else run['public_release']['mode'],
                  source='archived-production' if saved else 'new-rerun', confidence_level=.95)
    if study in ('euclidean', 'squared'):
        frame = pd.read_csv(source/'parent_summaries.csv.gz', float_precision='round_trip')
        panel = 'sample_size' if experiment == 'size-sweep' else 'epsilon'
        frame = frame[frame.panel == panel]
        group = ['dataset', 'n', 'd', 'objective_p', 'mechanism', 'epsilon']
        expected = 20 if saved else cfg['parents']
        partitions = 5 if saved else cfg['partitions']
        repetitions = 50 if saved else cfg['K']
        if frame.duplicated(group+['parent']).any() or not (frame.partition_count == partitions).all():
            raise ValueError('Duplicated parent or missing partition average')
        rows = []
        for keys, part in frame.groupby(group, sort=True):
            if len(part) != expected or set(part.parent) != set(range(expected)):
                raise ValueError('Missing or duplicate independent parents')
            mean, low, high = interval(part.mean_total_regret)
            row = dict(zip(group, keys))
            row.update(common, mean=mean, ci_low=low, ci_high=high, metric='total_excess_cost',
                parents=expected, partitions=partitions, chains=1, repetitions=repetitions,
                random_reference=float(part.random_minus_optimum.mean()),
                nonprivate_reference=0., theory_bound=float(part.expected_bound.mean()))
            rows.append(row)
        return rows
    if study == 'fashion':
        frame = pd.read_csv(source/'instance_summary.csv.gz', float_precision='round_trip')
        expected = 20 if saved else cfg['parents']
        partitions, chains = (5, 5) if saved else (cfg['partitions'], cfg['chains'])
        repetitions = 50 if saved else cfg['K']
        keys = ['parent', 'chain', 'd', 'mechanism']
        if frame.duplicated(keys+['partition']).any() or not (frame.groupby(keys).size() == partitions).all():
            raise ValueError('Missing or duplicate Fashion partitions')
        metrics = ['regret', 'RAND_gap']
        chain = frame.groupby(keys)[metrics].mean().reset_index()
        if not (chain.groupby(['parent', 'd', 'mechanism']).size() == chains).all():
            raise ValueError('Missing Fashion chains')
        parents = chain.groupby(['parent', 'd', 'mechanism'])[metrics].mean().reset_index()
        rows = []
        n, epsilon = (800, 16.) if saved else (cfg['n'], cfg['epsilon'])
        for (d, mechanism), part in parents.groupby(['d', 'mechanism'], sort=True):
            if len(part) != expected:
                raise ValueError('Missing Fashion parents')
            mean, low, high = interval(part.regret)
            rows.append(dict(common, dataset='mnist/fashion10k', n=n, d=d, objective_p=1,
                mechanism=mechanism, epsilon=epsilon, mean=mean, ci_low=low, ci_high=high,
                metric='total_excess_cost', parents=expected, partitions=partitions, chains=chains,
                repetitions=repetitions, random_reference=float(part.RAND_gap.mean()),
                nonprivate_reference=0., theory_bound=float(2*n*min(d, np.log(n))/epsilon)))
        return rows
    frame = pd.read_csv(source/'draws.csv.gz', float_precision='round_trip')
    if saved:
        with np.load(ROOT/'inputs/sonar/primary_null.npz') as z:
            baseline = float(z['nonprivate_pvalue'])
    else:
        baseline = read_json(source/'baseline.json')['nonprivate_p']
    expected = 1000 if saved else cfg['K']
    aliases = dict(ambient_radial='original_points', span_radial='projected_representation',
                   cost_infinity='distance_vectors', gumbel='gumbel')
    if frame.duplicated(['mechanism', 'epsilon', 'draw']).any():
        raise ValueError('Duplicated Sonar draws')
    rows = []
    for (mechanism, epsilon), part in frame.groupby(['mechanism', 'epsilon'], sort=True):
        if len(part) != expected or set(part.draw) != set(range(expected)):
            raise ValueError('Missing Sonar draws')
        mean, low, high = interval(part.primary_p)
        rows.append(dict(common, dataset='uci/sonar', n=208, d=60, objective_p=2,
            mechanism=aliases[mechanism], epsilon=epsilon, mean=mean, ci_low=low, ci_high=high,
            metric='mean_pvalue', parents=1, partitions=1, chains=1, repetitions=expected,
            random_reference=.5, nonprivate_reference=baseline, theory_bound=None))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--saved', action='store_true')
    sources.add_argument('--input', type=Path)
    parser.add_argument('--experiment', choices=('all', 'benchmarks', 'dimension-sweep', 'size-sweep', 'sonar'), required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    configure()
    import pandas as pd
    selections = dict(benchmarks=('euclidean', 'squared'), **{'dimension-sweep':('fashion',),
                      'size-sweep':('euclidean',), 'sonar':('sonar',)})
    experiments = list(selections) if args.experiment == 'all' else [args.experiment]
    rows = []
    for experiment in experiments:
        for study in selections[experiment]:
            path = ROOT/'saved'/study if args.saved else args.input/study/'aggregates'
            if not path.exists() and not args.saved and experiment == 'benchmarks':
                continue  # The command may request just one objective.
            rows.extend(summarize(study, path, experiment=experiment, saved=args.saved))
    if not rows:
        raise ValueError('No completed experiments found')
    frame = pd.DataFrame(rows)
    frame['mean_per_pair'] = frame['mean']/frame['n']
    frame['ci_low_per_pair'] = frame['ci_low']/frame['n']
    frame['ci_high_per_pair'] = frame['ci_high']/frame['n']
    frame.loc[frame.metric == 'mean_pvalue', ['mean_per_pair','ci_low_per_pair','ci_high_per_pair']] = float('nan')
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output/'summary.csv', index=False)
    write_json(args.output/'summary_metadata.json', dict(
        experiments=experiments, rows=len(frame), saved_results=args.saved,
        stochastic_experiments_executed=False,
        inference='Noise means -> partition means -> chain means (Fashion) -> independent parent means; Sonar uses draws',
        confidence='pointwise Student-t 95%; quick runs are demonstrations, not paper estimates'))
    print(f'Wrote {len(frame)} rows to {args.output / "summary.csv"}')


if __name__ == '__main__':
    main()
