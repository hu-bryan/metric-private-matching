"""Recompute the paper's numerical outputs; never sample mechanism noise."""
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys

from common import ROOT, DATA, configure, environment, frozen, read_json, write_json

METRICS = ['mean_total_cost','mean_total_regret','optimum','random_expected','random_minus_optimum','expected_bound']
FASHION_METRICS = ['regret','regret_per_pair','gap_fraction','OPT','RAND_gap','public_rank',
                   'effective_rank','projected_RMS','retained_energy','total_operator_norm','input_RMS']
ARMS = ['gumbel','original_points','projected_representation','distance_vectors']
SONAR_ARMS = dict(gumbel='Gumbel', ambient_radial='Original points',
                  span_radial='Projected representation', cost_infinity='Distance vectors')


def csv(path):
    import pandas as pd
    return pd.read_csv(path, float_precision='round_trip', low_memory=False)


def summarize_benchmark(parents):
    # Use the original aggregation implementation, including Student-t inference.
    from private_matching.clustering_manuscript_report import summarized
    keys = ['setting_id','panel','dataset','n','d','parent','objective_p','mechanism','epsilon']
    if parents.duplicated(keys).any() or not (parents.partition_count == 5).all():
        raise RuntimeError('Duplicated parent or missing partition average')
    counts = parents.groupby([k for k in keys if k != 'parent']).size()
    if not (counts == 20).all():
        raise RuntimeError('A benchmark setting is missing independent parents')
    return summarized(parents)


def summarize_fashion(frame):
    import numpy as np
    import pandas as pd
    from scipy.stats import t
    keys = ['parent','chain','d','mechanism']
    if frame.duplicated(keys+['partition']).any() or not (frame.groupby(keys).size() == 5).all():
        raise RuntimeError('Fashion partition hierarchy differs')
    chain = frame.groupby(keys)[FASHION_METRICS].mean().reset_index()
    if not (chain.groupby(['parent','d','mechanism']).size() == 5).all():
        raise RuntimeError('Fashion chain hierarchy differs')
    parents = chain.groupby(['parent','d','mechanism'])[FASHION_METRICS].mean().reset_index()
    rows = []
    for (d, arm), group in parents.groupby(['d','mechanism']):
        if len(group) != 20:
            raise RuntimeError('Fashion requires twenty independent parent means')
        row = dict(d=d, mechanism=arm, parents=len(group))
        for metric in FASHION_METRICS:
            mean = float(group[metric].mean())
            se = float(group[metric].std(ddof=1)/np.sqrt(len(group)))
            half = float(t.ppf(.975, len(group)-1)*se)
            row.update({metric:mean, metric+'_low':mean-half, metric+'_high':mean+half})
        rows.append(row)
    return pd.DataFrame(rows), parents, chain


def summarize_sonar(frame, baseline):
    import numpy as np
    import pandas as pd
    from scipy.stats import t
    if frame.duplicated(['mechanism','epsilon','draw']).any() or len(frame) != 20000:
        raise RuntimeError('Sonar needs 1000 draws for each of four methods and five budgets')
    rows = []
    for (arm, epsilon), group in frame.groupby(['mechanism','epsilon']):
        if len(group) != 1000 or set(group.draw) != set(range(1000)):
            raise RuntimeError('Sonar repetition IDs differ')
        values = group.sort_values('draw').primary_p.to_numpy()
        mean = float(values.mean())
        half = float(t.ppf(.975, 999)*values.std(ddof=1)/np.sqrt(1000))
        rows.append(dict(mechanism=SONAR_ARMS[arm], source_mechanism=arm, epsilon=epsilon,
            K=1000, mean_p=mean, mean_p_low_95=mean-half, mean_p_high_95=mean+half,
            nonprivate_p=baseline['nonprivate_p']))
    return pd.DataFrame(rows)


def calibration():
    import numpy as np
    from scipy.spatial.distance import pdist
    values = np.loadtxt(DATA/'sonar/sonar.data.gz')
    distances = pdist(values)
    median = float(np.median(distances))
    return dict(sonar_distance_min=float(distances.min()), sonar_distance_median=median,
        sonar_distance_max=float(distances.max()),
        swap_radius_at_epsilon16=float(np.log(3)/32),
        swap_epsilon_at_sonar_median=float(np.log(3)/(2*median)),
        factor_two_distance_epsilon8=float(np.log(2)/8),
        factor_two_distance_epsilon16=float(np.log(2)/16),
        log_800=float(np.log(800)), exp_4=float(np.exp(4)))


def compare(actual, expected, keys, name):
    import numpy as np
    import pandas as pd
    left = actual.sort_values(keys).reset_index(drop=True)
    right = expected.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right):
        return dict(name=name, passed=False, reason=f'row counts: {len(left)} vs {len(right)}')
    try:
        pd.testing.assert_frame_equal(left[keys], right[keys], check_dtype=False)
    except AssertionError:
        return dict(name=name, passed=False, reason='row identities differ')
    checked, failures, largest = [], [], 0.
    for column in left.columns.intersection(right.columns):
        if column in keys or not pd.api.types.is_numeric_dtype(left[column]):
            continue
        a, b = left[column].to_numpy(float), right[column].to_numpy(float)
        checked.append(column)
        difference = np.abs(a-b)
        if np.isfinite(difference).any():
            largest = max(largest, float(np.nanmax(difference)))
        if not np.allclose(a, b, rtol=5e-12, atol=5e-10, equal_nan=True):
            failures.append(column)
    return dict(name=name, passed=not failures, checked_rows=len(left), checked_columns=checked,
                failed_columns=failures, max_absolute_difference=largest)


def prose_numbers(combined, fashion, sonar, baseline, values, out):
    import numpy as np
    import pandas as pd
    rows = []
    def record(name, value, decimals, expected, source):
        printed = f'{float(value):.{decimals}f}'
        known = (name == 'uci/ecoli:gumbel:objective2:epsilon8' and expected == '36.54'
                 and abs(float(value)-36.534614267231305) < 5e-10)
        rows.append(dict(quantity=name, value=float(value), printed=printed,
                         expected=expected, matches=printed==expected, source=source,
                         known_manuscript_discrepancy=known and printed!=expected))
    def bench(dataset, arm, p):
        return combined[(combined.dataset==dataset)&(combined.mechanism==arm)&
                        (combined.objective_p==p)&(combined.epsilon==8)].mean_total_regret_mean.item()
    for dataset, arm, p, expected in (
        ('uci/wine','original_points',1,'16.43'), ('uci/wine','gumbel',1,'17.27'),
        ('uci/wdbc','gumbel',1,'16.48'), ('uci/wdbc','original_points',1,'21.26'),
        ('uci/ecoli','projected_representation',2,'14.92'), ('uci/ecoli','gumbel',2,'36.54')):
        record(f'{dataset}:{arm}:objective{p}:epsilon8', bench(dataset,arm,p), 2, expected,
               'numerical_experiment_summary.csv')
    for arm, d, expected in (('original_points',4,'76.4'),('original_points',104,'378.4'),
                              ('gumbel',4,'169.0'),('gumbel',104,'183.1')):
        record(f'fashion:{arm}:d{d}', fashion[(fashion.mechanism==arm)&(fashion.d==d)].regret.item(),
               1, expected, 'numerical_fashion_summary.csv')
    for arm, epsilon, expected in (('Gumbel',4,'0.4290'),('Gumbel',8,'0.2386'),('Gumbel',16,'0.0463'),
        ('Projected representation',16,'0.2539'),('Original points',16,'0.2924'),('Distance vectors',16,'0.3791')):
        record(f'sonar:{arm}:epsilon{epsilon}', sonar[(sonar.mechanism==arm)&(sonar.epsilon==epsilon)].mean_p.item(),
               4, expected, 'sonar_mean_pvalue.csv')
    record('sonar:nonprivate_T',baseline['nonprivate_T'],3,'7.064','Sonar non-private matching and rank-energy statistic')
    record('sonar:nonprivate_p',baseline['nonprivate_p'],4,'0.0022','Sonar primary null sample')
    record('sonar:public_rank',baseline['public_rank'],0,'47','Sonar reference grid')
    for name, decimals, expected in (
        ('sonar_distance_min',4,'0.1746'),('sonar_distance_median',4,'1.7786'),('sonar_distance_max',4,'3.5295'),
        ('swap_radius_at_epsilon16',4,'0.0343'),('swap_epsilon_at_sonar_median',3,'0.309'),
        ('factor_two_distance_epsilon8',4,'0.0866'),('factor_two_distance_epsilon16',4,'0.0433'),
        ('log_800',2,'6.68'),('exp_4',2,'54.60')):
        record(name,values[name],decimals,expected,'calibration.json')
    point = fashion[fashion.mechanism=='original_points'].set_index('d').regret
    gumbel = fashion[fashion.mechanism=='gumbel'].set_index('d').regret
    crossings = [(a,b) for a,b in zip(sorted(point.index)[:-1], sorted(point.index)[1:])
                 if point[a] < gumbel[a] and point[b] > gumbel[b]]
    if (19,24) not in crossings:
        raise RuntimeError('The reported Fashion crossing between d=19 and 24 is absent')
    f = fashion[(fashion.mechanism=='gumbel')&(fashion.d<=104)]
    bound = 2*800*np.minimum(f.d,np.log(800))/16
    through = int(f.loc[bound<f.RAND_gap,'d'].max())
    record('fashion:last_measured_d_with_bound_below_mean_random',through,0,'14','Fashion RAND_gap and selector bound')
    record('fashion:number_of_dimensions',fashion.d.nunique(),0,'56','Fashion dimension configuration')
    table = pd.DataFrame(rows)
    table.to_csv(out/'reported_numbers.csv', index=False)
    unexpected = ~table.matches & ~table.known_manuscript_discrepancy
    return dict(passed=not bool(unexpected.any()), exact_manuscript_agreement=bool(table.matches.all()),
                checked_quantities=len(table), failures=table.loc[unexpected].to_dict('records'),
                known_discrepancies=table.loc[table.known_manuscript_discrepancy].to_dict('records'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rerun', type=Path)
    args = parser.parse_args()
    configure()
    frozen('euclidean')
    import numpy as np
    import pandas as pd
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    figures = out/'figures'
    figures.mkdir(exist_ok=True)
    aggregate_root = out/'aggregates'
    aggregates = {}
    for study in ('euclidean','squared'):
        source = (args.rerun/study/'aggregates' if args.rerun else ROOT/'saved'/study)
        parents = csv(source/'parent_summaries.csv.gz')
        aggregates[study] = summarize_benchmark(parents)
        dest = aggregate_root/study
        dest.mkdir(parents=True, exist_ok=True)
        aggregates[study].to_csv(dest/'summary.csv', index=False)
    source = args.rerun/'fashion/aggregates' if args.rerun else ROOT/'saved/fashion'
    fashion, parents, chains = summarize_fashion(csv(source/'instance_summary.csv.gz'))
    dest = aggregate_root/'fashion'
    dest.mkdir(parents=True, exist_ok=True)
    fashion.to_csv(dest/'summary.csv', index=False)
    parents.to_csv(dest/'parent_summary.csv', index=False)
    chains.to_csv(dest/'chain_summary.csv', index=False)
    if args.rerun:
        baseline = read_json(args.rerun/'sonar/aggregates/baseline.json')
        source = args.rerun/'sonar/aggregates/draws.csv.gz'
    else:
        geom = read_json(ROOT/'inputs/sonar/geometry.json')
        with np.load(ROOT/'inputs/sonar/primary_null.npz') as z:
            p = float(z['nonprivate_pvalue'])
        baseline = dict(nonprivate_T=geom['nonprivate_T'], nonprivate_p=p, public_rank=geom['public_rank'])
        source = ROOT/'saved/sonar/draws.csv.gz'
    sonar = summarize_sonar(csv(source), baseline)
    sonar.to_csv(figures/'sonar_mean_pvalue.csv', index=False)
    values = calibration()
    write_json(out/'calibration.json', values)
    write_json(out/'execution_environment.json', environment())
    commands = [
        ['numerical.py', aggregate_root/'squared', figures, aggregate_root/'euclidean', aggregate_root/'fashion'],
        ['compact_bars.py', figures/'numerical_experiment_summary.csv', figures],
        ['sonar.py', figures], ['calibration.py', figures]]
    for command in commands:
        subprocess.run([sys.executable, str(ROOT/'renderers'/command[0]), *map(str, command[1:])], check=True)
    generated = (figures/'experiment_numerical_tables.tex').read_text()
    (out/'experiment_numerical_tables.tex').write_text(generated)
    combined = csv(figures/'numerical_experiment_summary.csv')
    reference = csv(ROOT/'reference/data/numerical_experiment_summary.csv')
    reference = reference[reference.panel.isin(['epsilon','sample_size'])]
    checks = [compare(combined, reference, ['panel','dataset','n','d','objective_p','mechanism','epsilon'], 'benchmark and Banknote values'),
              compare(fashion, csv(ROOT/'reference/data/numerical_fashion_summary.csv'), ['d','mechanism'], 'Fashion values'),
              compare(sonar, csv(ROOT/'reference/data/sonar_mean_pvalue.csv'), ['mechanism','epsilon'], 'Sonar mean p-values')]
    # Compare the printed rows, not TeX whitespace or float placement commands.
    def printed_rows(text):
        return [re.sub(r'\s+', '', line) for line in text.splitlines()
                if ' & ' in line and re.search(r'\d+\.\d{3}', line)]
    tables_match = printed_rows(generated) == printed_rows((ROOT/'reference/experiment_numerical_tables.tex').read_text())
    prose = prose_numbers(combined, fashion, sonar, baseline, values, out)
    # This enumerates all empirical figures actually referenced by this paper snapshot.
    names = read_json(ROOT/'provenance/staging.json')['reference_figures']
    present = all((figures/name).is_file() for name in names)
    passed = all(c['passed'] for c in checks) and tables_match and prose['passed'] and present
    status = ('passed_with_known_discrepancy' if prose['known_discrepancies'] else 'passed') if passed else 'different'
    report = dict(status=status, mode='rerun' if args.rerun else 'saved-results',
        comparisons=checks, all_printed_table_rows_match=tables_match,
        reported_numbers=prose, all_eight_paper_figures_present=present,
        stochastic_experiments_executed_by_this_command=False,
        pdf_bytes_compared=False, numeric_tolerance=dict(rtol=5e-12,atol=5e-10))
    write_json(out/'comparison.json', report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit('Outputs were written; some values differ. See comparison.json.')


if __name__ == '__main__':
    main()
