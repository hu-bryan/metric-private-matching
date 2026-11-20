"""Audited streaming export, parent-level inference, and readable benchmark report."""
from __future__ import annotations

import csv
import gzip
import itertools
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t

from .clustering_benchmark import (ARMS, NAME, NORMALIZATION, all_units, atomic_json,
                                   checkpoint, sha256)
from .integral_report import COLORS, LABELS, basic_html
from .login_guard import refuse_login_compute


def write_csv(path, rows, fields=None):
    path = Path(path)
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        if fields:
            path.write_text(','.join(fields)+'\n')
        return
    opener = gzip.open if path.suffix == '.gz' else open
    tmp = Path(str(path)+'.tmp')
    with opener(tmp, 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first) if fields is None else fields)
        writer.writeheader(); writer.writerow(first); writer.writerows(rows)
    os.replace(tmp, path)


def interval(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(mean=None, low=None, high=None, se=None, parent_count=0)
    se = float(a.std(ddof=1)/np.sqrt(len(a))) if len(a) > 1 else None
    half = float(t.ppf(.975, len(a)-1))*se if se is not None else None
    return dict(mean=float(a.mean()), low=float(a.mean()-half) if half is not None else None,
                high=float(a.mean()+half) if half is not None else None, se=se, parent_count=len(a))


def summarize(frame, group_columns, value):
    rows = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(group_columns, key), **{value+'_'+k: v for k, v in interval(group[value]).items()}))
    return pd.DataFrame(rows)


def validate(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if not Path(manifest['source_data']).exists():
        durable_source = Path(manifest['source_data'].replace('/vast/', '/ix1/', 1))
        if durable_source.exists():
            manifest['source_data'] = str(durable_source)
    if manifest['kind'] != NAME:
        raise RuntimeError('wrong benchmark kind')
    expected = all_units(manifest['design'])
    if len(list((directory/'units').glob('*.json'))) != len(expected):
        raise RuntimeError('STOP: incomplete/unexpected benchmark units')
    for file, rec in manifest['sources']['files'].items():
        if sha256(Path(manifest['source_data'])/file) != rec['sha256']:
            raise RuntimeError('STOP: source checksum changed: '+file)
    for unit in expected:
        if checkpoint(directory, unit, manifest['fingerprint'], verify=True) is None:
            raise RuntimeError('STOP: missing checkpoint: '+unit['instance_id'])
    return manifest, expected


def common_row(directory, meta, manifest):
    unit = meta['unit']
    q, r = meta['diagnostics'].get('Q', {}), meta['diagnostics'].get('R', {})
    return dict(study=NAME, fingerprint=manifest['fingerprint'], phase=manifest['design']['phase'],
                instance_id=unit['instance_id'], setting_id=unit['setting_id'], dataset=unit['dataset'],
                condition=unit['condition'], panel_memberships=';'.join(x['panel'] for x in unit['memberships']),
                parent=unit['parent'], partition=unit['partition'], n=unit['n'], d=unit['d'],
                natural_log_n=float(np.log(unit['n'])), log2_n=float(np.log2(unit['n'])),
                d_over_n=unit['d']/unit['n'], d_over_log_n=unit['d']/np.log(unit['n']),
                feature_order=unit['feature_order'], normalization_protocol=NORMALIZATION,
                source_pooled_rms_radius=meta['source_radius'], standardized_pooled_rms_radius=1. if not meta['degenerate'] else None,
                standardized_privacy_metric='sum_record_euclidean',
                privacy_units='standardized_input_only_no_raw_preprocessing_claim',
                q_rank=q.get('covariance_rank'), r_rank=r.get('covariance_rank'),
                q_effective_rank=q.get('effective_rank'), r_effective_rank=r.get('effective_rank'),
                q_rank_ratio=q.get('effective_rank_over_rank'), r_rank_ratio=r.get('effective_rank_over_rank'),
                q_centered_rms=q.get('centered_rms_radius'), r_centered_rms=r.get('centered_rms_radius'),
                q_noise_proportion=q.get('noise_proportion'), r_noise_proportion=r.get('noise_proportion'),
                side_center_separation=meta.get('cross_distance_summary', {}).get('side_center_separation'),
                cross_distance_mean=meta.get('cross_distance_summary', {}).get('mean'),
                cross_distance_median=meta.get('cross_distance_summary', {}).get('median'),
                geometry_file=f'../units/{unit["instance_id"]}.geometry.npz',
                metadata_file=f'../units/{unit["instance_id"]}.json',
                permutation_file=f'../units/{unit["instance_id"]}.npz')


def export(directory, output):
    """No simulation; export one phase and fingerprint, without duplicated aliases."""
    refuse_login_compute('export clustering benchmark results')
    directory, output = Path(directory), Path(output)
    manifest, expected = validate(directory)
    output.mkdir(parents=True, exist_ok=True)
    cfg = manifest['design']
    master = output/'analysis_table.csv.gz'
    tmp = Path(str(master)+'.tmp')
    instances, geometry_rows, runtimes, alias_rows = [], [], [], []
    raw_rows = 0; solves = 0; degenerate = 0
    writer = None
    started = time.perf_counter()
    with gzip.open(tmp, 'wt', newline='', compresslevel=3) as stream:
        for number, unit in enumerate(expected):
            stem = directory/'units'/unit['instance_id']
            meta = json.loads(Path(str(stem)+'.json').read_text())
            common = common_row(directory, meta, manifest)
            solves += meta['actual_solves']
            runtimes.append(dict(instance_id=unit['instance_id'], n=unit['n'], d=unit['d'],
                                 elapsed_seconds=meta['elapsed_seconds'], actual_solves=meta['actual_solves'],
                                 process_peak_rss_kib=meta.get('process_peak_rss_kib'),
                                 preparation_seconds=meta.get('preparation_seconds'), host=meta['host'], job_id=meta['job_id']))
            for membership in unit['memberships']:
                alias_rows.append(dict(instance_id=unit['instance_id'], **membership,
                                       alias=len(unit['memberships']) > 1, no_duplicate_execution=True))
            if meta['degenerate']:
                degenerate += 1
                for p in (1, 2):
                    geometry_rows.append(dict(common, objective_p=p, optimum_total_cost=0., random_total_cost=0., gap=0.,
                                              gap_tolerance=0., gap_status='degenerate_zero_pooled_radius', status='degenerate'))
                continue
            with np.load(str(stem)+'.npz') as saved:
                values = {key: saved[key] for key in ('total_cost', 'total_regret', 'row_oscillation', 'realization_bound', 'runtime_seconds', 'solve_seconds')}
                for ip, p in enumerate((1, 2)):
                    optimum = meta['optima'][str(p)]
                    random = meta['random_expected'][str(p)]
                    gap = meta['gaps'][str(p)]
                    geom = dict(common, objective_p=p, optimum_total_cost=optimum, random_total_cost=random,
                                gap=gap['value'], gap_tolerance=gap['tolerance'], gap_status=gap['status'], status='complete')
                    geometry_rows.append(geom)
                    for ia, arm in enumerate(ARMS[p]):
                        reuse = meta['reuse'][f'{p}:{arm}']
                        for ie, epsilon in enumerate(cfg['epsilon']):
                            cert = meta['bounds'][f'{p}:{arm}'][ie]
                            costs = values['total_cost'][ip, ia, ie]
                            regrets = values['total_regret'][ip, ia, ie]
                            score = float(regrets.mean()/gap['value']) if gap['status'] != 'zero_gap' else None
                            summary = dict(geom, mechanism=arm, epsilon=epsilon, mean_total_cost=float(costs.mean()),
                                           mean_total_regret=float(regrets.mean()), score=score,
                                           noise_sd_total_regret=float(regrets.std(ddof=1)) if len(regrets)>1 else None,
                                           noise_se_total_regret=float(regrets.std(ddof=1)/np.sqrt(len(regrets))) if len(regrets)>1 else None,
                                           bound_total=cert['bound_total'], expected_bound=True,
                                           conditional_noise_mean_over_bound=float(regrets.mean()/cert['bound_total']) if cert['bound_total'] else None,
                                           K=cfg['K'], source_arm=reuse['source_arm'], reuse_reason=reuse['reuse_reason'])
                            instances.append(summary)
                            base = dict(geom, mechanism=arm, epsilon=epsilon,
                                        original_units_rate_if_transform_frozen=epsilon/meta['source_radius'],
                                        raw_cost_multiplier=meta['source_radius']**p,
                                        privacy_output='final_matching_only' if arm == 'gumbel' else 'local_report_and_matching',
                                        source_arm=reuse['source_arm'], reuse_reason=reuse['reuse_reason'],
                                        source_objective=p if reuse['source_arm'] else None,
                                        unit_noise_protocol='shared_unit_draw_across_epsilon_and_supported_objectives',
                                        bound_total=cert['bound_total'], bound_uncapped=cert['bound_uncapped'],
                                        bound_cap=cert['bound_cap'], bound_id=cert['bound_id'],
                                        epsilon_y=cert['epsilon_y'], epsilon_s=cert['epsilon_s'],
                                        L_infinity=cert['L_infinity'], L_one=cert['L_one'])
                            for draw in range(cfg['K']):
                                row = dict(base, noise_draw=draw,
                                           noise_coupling_id=f'{cfg["phase"]}:{unit["instance_id"]}:unit:{draw}:{reuse["source_arm"] or arm}',
                                           total_cost=float(costs[draw]), total_regret=float(regrets[draw]),
                                           normalized_score=float(regrets[draw]/gap['value']) if gap['status'] != 'zero_gap' else None,
                                           row_oscillation=float(values['row_oscillation'][ip, ia, ie, draw]),
                                           realization_bound=float(values['realization_bound'][ip, ia, ie, draw]),
                                           runtime_seconds=float(values['runtime_seconds'][ip, ia, ie, draw]),
                                           solve_seconds=float(values['solve_seconds'][ip, ia, ie, draw]),
                                           permutation_array='permutations', permutation_index=f'{ip},{ia},{ie},{draw}')
                                if writer is None:
                                    writer = csv.DictWriter(stream, fieldnames=list(row))
                                    writer.writeheader()
                                writer.writerow(row); raw_rows += 1
            if (number+1) % 100 == 0:
                print(f'EXPORT {number+1}/{len(expected)} rows={raw_rows}', flush=True)
    os.replace(tmp, master)
    df = pd.DataFrame(instances)
    write_csv(output/'instance_summaries.csv.gz', instances)
    write_csv(output/'geometry_table.csv.gz', geometry_rows)
    write_csv(output/'runtime_by_instance.csv', runtimes)
    write_csv(output/'panel_aliases.csv', alias_rows)
    # Expand memberships only for panel-specific summaries. There is one raw execution.
    aliases = pd.DataFrame(alias_rows).rename(columns={'condition':'panel_condition'})
    expanded = df.merge(aliases[['instance_id','panel','panel_condition']], on='instance_id', validate='many_to_many')
    keys = ['panel','setting_id','dataset','condition','n','d','objective_p','mechanism','epsilon']
    metrics = ['score','mean_total_regret','mean_total_cost','bound_total','conditional_noise_mean_over_bound',
               'q_rank','r_rank','q_effective_rank','r_effective_rank','q_rank_ratio','r_rank_ratio','gap','source_pooled_rms_radius']
    parent = expanded.groupby(keys+['parent'], dropna=False, sort=True)[metrics].mean().reset_index()
    parent['partition_count'] = expanded.groupby(keys+['parent'], dropna=False).size().values
    parent.to_csv(output/'parent_summaries.csv.gz', index=False)
    summaries = None
    for metric in metrics:
        part = summarize(parent, keys, metric)
        summaries = part if summaries is None else summaries.merge(part, on=keys, validate='one_to_one')
    summaries.to_csv(output/'summary.csv', index=False)
    # Equal weights for six datasets. The index identifies independent draws of
    # a six-dataset product collection; it does not make source corpora random.
    cross = parent[parent.panel == 'aggregate'].groupby(['parent','objective_p','mechanism','epsilon'])[metrics].mean().reset_index()
    cross_summary = None
    for metric in ('score','mean_total_regret','mean_total_cost'):
        part = summarize(cross, ['objective_p','mechanism','epsilon'], metric)
        cross_summary = part if cross_summary is None else cross_summary.merge(part, on=['objective_p','mechanism','epsilon'])
    cross.to_csv(output/'equal_dataset_parent_summaries.csv', index=False)
    cross_summary.to_csv(output/'equal_dataset_summary.csv', index=False)
    paired, paired_parents = paired_comparisons(parent)
    write_csv(output/'paired_comparisons.csv', paired)
    write_csv(output/'paired_parent_differences.csv.gz', paired_parents)
    associations = rank_associations(expanded, cfg['association_epsilon'])
    write_csv(output/'rank_associations.csv', associations)
    write_csv(output/'geometry_range_comparisons.csv', geometry_range_comparisons(expanded))
    checks = dict(fingerprint=manifest['fingerprint'], expected_instances=len(expected), committed_instances=len(expected),
                  degenerate_instances=degenerate, canonical_draw_rows=raw_rows,
                  expected_draw_rows=(len(expected)-degenerate)*2*7*len(cfg['epsilon'])*cfg['K'],
                  actual_assignment_solves=solves, parent_collections=cfg['parents'], partitions_per_parent=cfg['partitions'],
                  noise_draws_per_arm_budget=cfg['K'], budgets=cfg['epsilon'],
                  source_checksums_verified=True, checkpoint_checksums_verified=True,
                  canonical_sha256=sha256(master), exporter_seconds=time.perf_counter()-started,
                  worker_seconds=sum(x['elapsed_seconds'] for x in runtimes),
                  small_positive_gap_objectives=sum(g['gap_status']=='small_positive_gap' for g in geometry_rows),
                  zero_gap_objectives=sum(g['gap_status']=='zero_gap' for g in geometry_rows))
    if raw_rows != checks['expected_draw_rows']:
        raise RuntimeError('STOP: canonical completeness mismatch')
    atomic_json(output/'verification.json', checks)
    atomic_json(output/'manifest.json', manifest)
    # Per-objective exact OPT and RAND references are in the geometry table;
    # exact optimum permutations remain in each outcome archive.
    dictionary = {key: field_description(key) for key in list(common_row(directory, json.loads((directory/'units'/(expected[0]['instance_id']+'.json')).read_text()), manifest))}
    dictionary.update({key: field_description(key) for key in (writer.fieldnames if writer else [])})
    atomic_json(output/'data_dictionary.json', dictionary)
    (output/'data_dictionary.md').write_text(dictionary_text())
    return manifest, summaries, cross_summary, parent, expanded, checks


def paired_comparisons(parent):
    results, raw = [], []
    keys = ['panel','setting_id','dataset','condition','n','d','objective_p','epsilon']
    for key, group in parent.groupby(keys, sort=True):
        common = dict(zip(keys, key))
        p = common['objective_p']
        # All requested applicable pairs; differences use the same parent draws.
        for a, b in itertools.combinations(ARMS[p], 2):
            left = group[group.mechanism == a].set_index('parent')
            right = group[group.mechanism == b].set_index('parent')
            if not left.index.equals(right.index):
                raise RuntimeError('parent pairing mismatch')
            differences = left.score-right.score
            result = dict(common, mechanism_a=a, mechanism_b=b, difference='a minus b; negative favors a',
                          **{'score_difference_'+k: v for k,v in interval(differences).items()})
            result.update({'regret_difference_'+k: v for k,v in interval(left.mean_total_regret-right.mean_total_regret).items()})
            results.append(result)
            for parent_id, value in differences.items():
                raw.append(dict(common, mechanism_a=a, mechanism_b=b, parent=parent_id,
                                score_difference=None if not np.isfinite(value) else float(value)))
    return results, raw


def rank_associations(expanded, epsilon):
    rows = []
    # Describe within dataset/condition/setting at one fixed epsilon. Use parent
    # means for both differences and diagnostics; no pseudo replication.
    frame = expanded[expanded.epsilon == epsilon]
    context = ['panel','setting_id','dataset','condition','n','d','objective_p']
    metrics = ['q_rank','r_rank','q_effective_rank','r_effective_rank','q_rank_ratio','r_rank_ratio']
    for key, group in frame.groupby(context, sort=True):
        common = dict(zip(context, key), epsilon=epsilon)
        group = group.groupby(['parent','mechanism'])[['score']+metrics].mean().reset_index()
        for a, b in itertools.combinations(ARMS[common['objective_p']], 2):
            left = group[group.mechanism == a].set_index('parent')
            right = group[group.mechanism == b].set_index('parent')
            diff = left.score-right.score
            for metric in metrics:
                mask = np.isfinite(diff) & np.isfinite(left[metric])
                valid = int(mask.sum())
                variable = valid >= 3 and diff[mask].nunique() > 1 and left.loc[mask,metric].nunique() > 1
                rho = float(spearmanr(left.loc[mask,metric], diff[mask]).statistic) if variable else None
                rows.append(dict(common, mechanism_a=a, mechanism_b=b, diagnostic=metric, spearman_rho=rho,
                                 parents=valid, status='exploratory_within_setting' if variable else 'constant_or_insufficient',
                                 private_diagnostic='NONPRIVATE_EXPLANATION_ONLY' if metric.startswith('r_') else 'public_geometry'))
    return rows


def field_description(key):
    descriptions = {
        'normalized_score':'Per-draw total regret / exact RAND-minus-OPT for this instance. Means are computed within instance before aggregation; null for zero gap.',
        'source_pooled_rms_radius':'One scalar radius of actual raw Q and R jointly, after row and feature selection.',
        'original_units_rate_if_transform_frozen':'epsilon / construction radius; algebraic conversion only, NOT a guarantee for data-dependent raw preprocessing.',
        'raw_cost_multiplier':'Multiply standardized cost/regret by radius**objective_p to audit the same assignment in original coordinates.',
        'panel_memberships':'Semicolon-separated panels sharing this one execution. Never treat aliases as independent instances.',
        'noise_coupling_id':'Explicit unit-noise channel shared across epsilon and objectives; cross-arm equality only with source_arm reuse.',
        'permutation_index':'Comma-separated indices into permutation_array in permutation_file: objective index, arm index, epsilon index, draw index.',
        'realization_bound':'Per-realization certificate, checked during simulation; anchor uses its actual-anchor secant bound.',
        'bound_total':'Analytic expected total-regret upper bound for current standardized geometry; not a per-draw or empirical-mean ceiling.',
        'bound_cap':'Public Euclidean cap n*public diameter; squared cap unavailable (null), no synthetic-domain cap.',
        'q_rank':'Numerical centered covariance rank on actual standardized public Q; threshold and singular values in metadata_file.',
        'r_rank':'Numerical centered covariance rank on actual standardized private R; nonprivate explanatory diagnostic.',
        'geometry_file':'Path relative to handoff. NPZ contains standardized Q/R, raw normalization center/radius, source row IDs, labels, pixel IDs, parent pools and maximum lists.',
        'metadata_file':'Path relative to handoff. JSON contains complete spectra, rank tolerances, within-label diagnostics, expected bounds, checksums, timings and job IDs.',
    }
    return descriptions.get(key, key.replace('_',' ')+'; see the normalization, uncertainty, units and provenance rules in data_dictionary.md and manifest.json')


def dictionary_text():
    return '''# Benchmark numerical record

Costs and regrets are sums over all matched pairs. Objective 1 is Euclidean;
objective 2 is squared Euclidean. Q is public and R private; permutation[i]
is the Q destination assigned to private R[i]. All assignments are bijections.
The exact OPT permutation is outcome array optimum_permutations[objective_index].

The canonical gzip CSV has one row per private mechanism execution label/draw,
including explicit equivalent-arm reuse. Its geometry/metadata/permutation paths
are relative to this handoff. Exact OPT and exact RAND are in geometry_table.csv.gz.
No Monte Carlo random reference is substituted. panel_aliases.csv maps one actual
instance to its panel memberships; aliases do not supply independent evidence.

The actual selected Q and R are jointly centered and scaled to pooled RMS radius
one separately at every n and d. Inputs are standardized before mechanism execution.
Privacy is epsilon times the sum of Euclidean record displacements in those units.
No privacy claim applies to raw data-dependent normalization. The field
original_units_rate_if_transform_frozen is algebraic only. Raw cost conversion
multiplies by source_pooled_rms_radius**objective_p.

The fixed zero-gap tolerance is 64*float64_eps*max(1,abs(OPT),abs(RAND)).
Scores at gaps below this tolerance are null. Positive small gaps above it remain
included and are flagged using 1e-6*max(1,abs(OPT),abs(RAND)). Values above one remain.
Zero pooled-radius units remain in geometry/completeness tables with no private
draws. Zero-variance effective ranks are null. Participation-ratio effective rank
uses all sample-covariance eigenvalues, with covariance denominator n-1. Numerical
rank uses singular > largest_singular*max(n,d)*float64_eps; full spectra are saved.

Noise means are computed per instance; five partition means then form each parent
mean. Pointwise 95% Student-t intervals use the 20 parent means. They are neither
simultaneous nor adjusted for multiple comparisons. Different parent samples can
reuse source rows; inference is conditional on each fixed benchmark corpus.
Equal-dataset summaries give each of six datasets weight 1/6. Their intervals use
20 independent draws of the six-dataset product collection. Paired differences
use the same parent collections, never an assumption of shared mechanism noise.

All seven epsilon settings explicitly reuse each arm's unit noise. The same base
channel is shared across supported objectives. Gumbel protects only the final
matching; other arms protect reusable local reports. Equivalent full-dimensional
radial/fold/span endpoints reuse assignments and retain source-arm provenance.
Expected bounds are distinct from realized certificates. Rank relationships at
epsilon=4 are exploratory within setting and use parent means; private R diagnostics
are not inputs to tuning. All applicable mechanism pairs are exported; no selection
by significance or winner is performed.

Pilot and production seeds have different phase namespaces and fingerprints.
Source snapshots, dependency versions, pinned datasets and source checksums are
saved in the run directory. Resume validates complete checkpoints by SHA256.
Single-threaded numerical execution and pinned source/environment are required.
Wall-clock runtime and archive container timestamps are not deterministic scientific
outcomes. Reproduction compares arrays/permutations/costs, not runtime bytes.
'''


def readable_tables(output, summary, cross):
    tables = output/'tables'; tables.mkdir(exist_ok=True)
    links = []
    def table(frame, name, title, index):
        lines = ['# '+title, '', 'Mean normalized regret; 0 = OPT, 1 = exact RAND. Parent-level 95% intervals are in summary.csv.', '']
        p = int(frame.objective_p.iloc[0]); arms = ARMS[p]
        lines += ['| '+index+' | '+' | '.join(LABELS[a] for a in arms)+' |', '|---|'+'---:|'*len(arms)]
        for value in sorted(frame[index].unique()):
            part = frame[frame[index] == value].set_index('mechanism')
            lines.append('| '+str(value)+' | '+' | '.join('undefined' if a not in part.index or not np.isfinite(part.loc[a,'score_mean']) else f'{part.loc[a,"score_mean"]:.3f}' for a in arms)+' |')
        (tables/(name+'.md')).write_text('\n'.join(lines)+'\n')
        links.append((title, 'tables/'+name+'.md'))
    for p in (1,2):
        for setting, frame in summary[(summary.panel == 'aggregate') | (summary.panel == 'label')].groupby('setting_id'):
            frame = frame[frame.objective_p == p]
            table(frame, setting+f'_p{p}', setting+f', objective {p}, epsilon rows', 'epsilon')
        part = cross[cross.objective_p == p]
        table(part, f'equal_datasets_p{p}', f'Equal dataset weights, objective {p}, epsilon rows', 'epsilon')
        for panel, index in (('dimension','d'),('sample_size','n')):
            for epsilon in sorted(summary.epsilon.unique()):
                part = summary[(summary.panel == panel)&(summary.objective_p == p)&(summary.epsilon == epsilon)]
                table(part, f'{panel}_p{p}_e{epsilon:g}', f'{panel}, objective {p}, epsilon {epsilon:g}', index)
    return links


def plots(output, summary, cross, expanded, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False})
    figures = output/'figures'; figures.mkdir(exist_ok=True)
    links = []
    def save(fig, name, caption):
        fig.savefig(figures/(name+'.png'), dpi=145, bbox_inches='tight')
        fig.savefig(figures/(name+'.pdf'), bbox_inches='tight')
        plt.close(fig); links.append((name, caption))
    def draw(ax, frame, p, x, metric, reference=True):
        for arm in ARMS[p]:
            part = frame[frame.mechanism == arm].sort_values(x)
            if not len(part): continue
            mean = metric+'_mean'; low = metric+'_low'; high = metric+'_high'
            ax.plot(part[x], part[mean], color=COLORS[arm], label=LABELS[arm], marker='o', markersize=2.7, lw=1.2)
            if low in part and part[low].notna().any():
                ax.fill_between(part[x].to_numpy(float), part[low].to_numpy(float), part[high].to_numpy(float), color=COLORS[arm], alpha=.10)
        if reference:
            ax.axhline(1, color='#333333', ls='--', lw=.8)
            ax.axhline(0, color='#999999', lw=.6)
        ax.grid(alpha=.15); ax.set_xlabel(x)
    for p in (1,2):
        objective = 'Euclidean distance' if p == 1 else 'Squared Euclidean distance'
        agg = summary[(summary.panel=='aggregate')&(summary.objective_p==p)]
        fig, axes = plt.subplots(2,3, figsize=(13,7), sharex=True)
        for ax, (dataset, frame) in zip(axes.flat, agg.groupby('dataset')):
            draw(ax, frame, p, 'epsilon','score'); ax.set_xscale('log',base=2); ax.set_title(dataset); ax.set_ylabel('Normalized regret')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
        fig.suptitle(objective+' — six separate source corpora; shaded pointwise 95% parent intervals')
        fig.tight_layout(rect=(0,.09,1,.95)); save(fig, f'aggregate_p{p}', 'Each dataset keeps its own curve; 0 is optimal and 1 is the exact random reference. Scores are not clipped.')
        fig, ax = plt.subplots(figsize=(8,4.7))
        draw(ax, cross[cross.objective_p==p],p,'epsilon','score'); ax.set_xscale('log',base=2)
        ax.set_ylabel('Mean normalized regret'); ax.set_title(objective+' — equal weights for six datasets')
        ax.legend(ncol=2, fontsize=8); fig.tight_layout(); save(fig,f'equal_datasets_p{p}','Each dataset contributes weight 1/6, after noise, partition and parent aggregation.')
        fig, axes = plt.subplots(2,2,figsize=(11,7),sharex=True)
        label = summary[(summary.panel=='label')&(summary.objective_p==p)]
        for ax,(condition,frame) in zip(axes.flat,label.groupby('condition')):
            draw(ax,frame,p,'epsilon','score'); ax.set_xscale('log',base=2); ax.set_title(condition); ax.set_ylabel('Normalized regret')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False)
        fig.suptitle(objective+' — MNIST label-conditioned matching, n=64, d=784')
        fig.tight_layout(rect=(0,.09,1,.95)); save(fig,f'labels_p{p}','All four directions are reported separately. Each actual pair of inputs is jointly standardized.')
        for panel, x in (('dimension','d'),('sample_size','n')):
            for metric in ('score','mean_total_regret'):
                fig,axes=plt.subplots(2,4,figsize=(16,7.4))
                for ax,epsilon in zip(axes.flat,manifest['design']['epsilon']):
                    frame=summary[(summary.panel==panel)&(summary.objective_p==p)&(summary.epsilon==epsilon)]
                    draw(ax,frame,p,x,metric,metric=='score'); ax.set_xscale('log',base=2); ax.set_title(f'epsilon = {epsilon:g}')
                    ticks = [3,6,12,24,49,98,256,784] if panel=='dimension' else [16,32,64,128,256]
                    ax.set_xticks(ticks, labels=[str(v) for v in ticks])
                    ax.tick_params(axis='x', labelsize=7)
                    ax.set_ylabel('Normalized regret' if metric=='score' else 'Total regret')
                    if panel=='dimension':
                        ax.axvline(256,color='#777777',ls=':',lw=.8)
                        ax.axvline(np.log(256),color='#777777',ls='-.',lw=.8)
                for ax in axes.flat[len(manifest['design']['epsilon']):]: ax.axis('off')
                handles,labels=axes.flat[0].get_legend_handles_labels()
                fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False)
                fig.suptitle(f'{objective} — {panel}; fixed epsilon in each panel; per-setting pooled RMS = 1')
                fig.tight_layout(rect=(0,.09,1,.95))
                save(fig,f'{panel}_{metric}_p{p}','Nested source records/features before setting-specific standardization. Dimension references: n=256 (dotted), natural log(n)=5.55 (dash-dot); log2(n)=8 is retained in the numerical record.')
        # Expected bound comparisons at a prespecified epsilon, one point per setting/arm.
        frame=expanded[(expanded.objective_p==p)&(expanded.epsilon==4.)].drop_duplicates(['instance_id','mechanism'])
        means=frame.groupby(['setting_id','mechanism'])[['mean_total_regret','bound_total']].mean().reset_index()
        fig,ax=plt.subplots(figsize=(7,5))
        for arm in ARMS[p]:
            part=means[means.mechanism==arm]
            ax.scatter(part.bound_total,part.mean_total_regret,label=LABELS[arm],color=COLORS[arm],s=23,alpha=.8)
        positive=means[['mean_total_regret','bound_total']].to_numpy().ravel(); positive=positive[positive>0]
        if len(positive):
            lo,hi=positive.min()*.7,positive.max()*1.4; ax.plot([lo,hi],[lo,hi],color='black',ls='--',lw=.8)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Expected total-regret bound');ax.set_ylabel('Empirical mean total regret')
        ax.set_title(objective+' — bounds at epsilon=4');ax.legend(fontsize=7,ncol=2);fig.tight_layout()
        save(fig,f'bounds_p{p}','Empirical conditional means and analytic expected bounds differ from per-draw certificates. The diagonal is a reference, not a required pointwise ceiling on an empirical mean.')
        # Prespecified rank explanations at epsilon=4. Each point is a parent mean.
        pair=('anchor','terminal_fold') if p==1 else ('coordinate_allocated','span_radial')
        grouped=frame.groupby(['parent','setting_id','dataset','condition','mechanism'])[['score','q_effective_rank','r_effective_rank']].mean().reset_index()
        left=grouped[grouped.mechanism==pair[0]].drop(columns='mechanism')
        right=grouped[grouped.mechanism==pair[1]][['parent','setting_id','score']]
        joined=left.merge(right,on=['parent','setting_id'],suffixes=('_a','_b'))
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for dataset,part in joined.groupby('dataset'):
            for ax,metric in zip(axes,('q_effective_rank','r_effective_rank')):
                ax.scatter(part[metric],part.score_a-part.score_b,label=dataset,s=9,alpha=.25)
                ax.axhline(0,color='#777777',lw=.7); ax.set_xlabel(metric.replace('_',' ')); ax.set_ylabel('Paired normalized-score difference')
        axes[0].legend(fontsize=6);fig.suptitle(f'epsilon=4, objective {p}: {LABELS[pair[0]]} minus {LABELS[pair[1]]}')
        fig.tight_layout();save(fig,f'rank_explanation_p{p}','Exploratory parent-level paired differences. Colors identify source corpora; exact condition and setting identities are retained in rank_associations.csv. Private-side diagnostics are nonprivate explanations and never tune mechanisms.')
    # Source-label preview is reconstructed from saved pinned data, not a new draw.
    from .clustering_benchmark import load_dataset
    images,labels=load_dataset(manifest['source_data'],'mnist/digits10k')
    fig,axes=plt.subplots(2,8,figsize=(11,3))
    for row,digit in enumerate((6,9)):
        for ax,index in zip(axes[row],np.flatnonzero(labels==digit)[:8]):
            ax.imshow(images[index].reshape(28,28),cmap='gray');ax.set_title(f'row {index}: {digit}',fontsize=8);ax.axis('off')
    fig.tight_layout();save(fig,'mnist_label_validation','Stored labels 6 and 9 match semantic sixes and nines. All 10,000 stored labels were checked against the original MNIST test-label file; stored 10 means digit zero.')
    summary.to_csv(figures/'plot_input_summary.csv',index=False)
    return links


def report(output, manifest, summary, cross, checks, links, tables):
    cfg=manifest['design']
    def fmt(value): return 'undefined' if value is None or not np.isfinite(value) else f'{value:.3f}'
    lines=['# Clustering benchmark matching results','',
           f'Completed {cfg["phase"]} run `{manifest["fingerprint"]}`. This report describes the frozen design, not pilot evidence.' if cfg['phase']=='production' else f'Pilot smoke/resource record `{manifest["fingerprint"]}`. These draws are excluded from production science.', '',
           f'The study contains **{checks["expected_instances"]:,} distinct standardized instances**, {cfg["parents"]} parent collections per source/condition, {cfg["partitions"]} partitions per parent, {cfg["K"]} noise draws per private arm and budget, and {len(cfg["epsilon"])} epsilon values. It performed **{checks["actual_assignment_solves"]:,} exact assignment solves**.', '',
           'Each actual Q/R instance is jointly centered and scaled to pooled RMS radius one before any mechanism runs. Privacy is measured in these standardized coordinates. The data-dependent construction is an offline simulation step; no privacy guarantee is claimed for preprocessing raw confidential data.', '',
           'The main score is mean total regret divided by exact RAND minus OPT, calculated within each instance before averaging. **0 is optimal; 1 is the exact random-reference expected cost; values above 1 are retained.** Costs and absolute regrets are sums over all n matched pairs. Euclidean and squared objectives remain separate.', '',
           '## Aggregate comparisons','',
           'The following compact table fixes epsilon=4. Each best method is the smallest observed mean among the seven applicable private arms, a descriptive comparison rather than a multiple-comparison-adjusted claim. Confidence intervals are pointwise 95% intervals over parent means.', '',
           '| Dataset | Objective | Lowest observed mean | Score [95% interval] | Gumbel | Point radial |','|---|---|---|---:|---:|---:|']
    for p in (1,2):
        for dataset,frame in summary[(summary.panel=='aggregate')&(summary.objective_p==p)&(summary.epsilon==4.)].groupby('dataset'):
            best=frame.loc[frame.score_mean.idxmin()];by=frame.set_index('mechanism')
            lines.append(f'| {dataset} | {"Distance" if p==1 else "Squared"} | {LABELS[best.mechanism]} | {fmt(best.score_mean)} [{fmt(best.score_low)}, {fmt(best.score_high)}] | {fmt(by.loc["gumbel","score_mean"])} | {fmt(by.loc["ambient_radial","score_mean"])} |')
    lines += ['', 'Each of the six datasets receives equal weight in the cross-dataset summary. Parent selections are independent conditional on the fixed corpus and can reuse source rows; the public/private halves are complementary within each parent.', '']
    for p in (1,2):
        frame=cross[(cross.objective_p==p)&(cross.epsilon==4.)]
        best=frame.loc[frame.score_mean.idxmin()]
        lines.append(f'At epsilon=4, the lowest equal-dataset mean for objective {p} is **{LABELS[best.mechanism]}**, score **{fmt(best.score_mean)}** [{fmt(best.score_low)}, {fmt(best.score_high)}].')
    lines += ['', '![Six-source distance comparisons](figures/aggregate_p1.png)', '', '![Six-source squared comparisons](figures/aggregate_p2.png)', '',
              '## Same-label and different-label MNIST matching','',
              'All four directions use n=64 and d=784. Shared source lists enable paired comparisons; Q-to-R direction still matters because mechanisms use public Q geometry. Every condition has its own joint normalization.', '',
              '| Condition | Objective | Lowest observed mean at epsilon=4 | Score | Gumbel |','|---|---|---|---:|---:|']
    for p in (1,2):
        for condition,frame in summary[(summary.panel=='label')&(summary.objective_p==p)&(summary.epsilon==4.)].groupby('condition'):
            best=frame.loc[frame.score_mean.idxmin()]
            gum=frame[frame.mechanism=='gumbel'].score_mean.iloc[0]
            lines.append(f'| {condition} | {"Distance" if p==1 else "Squared"} | {LABELS[best.mechanism]} | {fmt(best.score_mean)} | {fmt(gum)} |')
    lines += ['', '![Label-conditioned distance curves](figures/labels_p1.png)', '', '![Label-conditioned squared curves](figures/labels_p2.png)', '',
              '## Dimension and sample-size sweeps','',
              'The dimension sweep fixes n=256 and deletes trailing coordinates of one publicly seeded pixel permutation. Its dimensions are 784, 392, 256, 196, 98, 49, 24, 12, 6 and 3. Dotted and dash-dot vertical references mark n=256 and natural log(n)=5.55. The base-two reference is log2(n)=8. These ambient references do not replace measured public rank.', '',
              'The sample-size sweep uses nested n=16,32,64,128,256 public/private six lists at d=784. Source records/features are nested before normalization. Radius is recomputed for every actual setting, so the standardized arrays need not be numerical prefixes. Both sweeps hold each plotted epsilon fixed.', '',
              'The n=64 label/sample-size endpoint and n=256,d=784 dimension/sample-size endpoint share their executions. All focused panels use the same saved pixel order, which permits the second alias without inferring a noise coupling from rotational or permutation invariance.', '']
    for panel in ('dimension','sample_size'):
        for p in (1,2):
            lines += [f'![{panel}, objective {p}, normalized scores](figures/{panel}_score_p{p}.png)', '',
                      f'[Absolute total-regret curves](figures/{panel}_mean_total_regret_p{p}.png)', '']
    lines += ['## Bounds and geometric explanations','',
              'Analytic expected bounds use the same standardized geometry and total-regret convention. No synthetic cube support cap is imported. Realization certificates, including the anchor secant bound, are checked for every draw; an expected bound is not treated as a per-realization ceiling.', '',
              'All applicable paired mechanism differences and their parent-level intervals are exported. Rank associations use epsilon=4 separately for each objective and setting, preserving dataset and condition identity. Correlations are exploratory; constant diagnostics have undefined correlations. Private-side ranks never enter mechanism selection or noise tuning.', '',
              '![Distance bounds](figures/bounds_p1.png)', '', '![Squared bounds](figures/bounds_p2.png)', '',
              '![Distance rank explanation](figures/rank_explanation_p1.png)', '', '![Squared rank explanation](figures/rank_explanation_p2.png)', '',
              '## Data and correctness checks','',
              f'Clustering Benchmarks v1.2.0 is pinned to commit `{manifest["sources"]["commit"]}`. Dataset matrices, labels0, descriptions and battery READMEs are preserved with SHA256 hashes. MNIST has 958 sixes and 1,009 nines. Stored digits 1–9 map to themselves; stored 10 is digit zero. The entire original test-label sequence validates this mapping.', '',
              'The initial checker incorrectly tested a plus-one label encoding; it failed before sampling. A row-by-row comparison with original MNIST labels and image previews established the correct mapping. No experimental setting or source dataset was changed.', '',
              f'Completeness: {checks["degenerate_instances"]} zero-radius instances; {checks["zero_gap_objectives"]} numerically zero assignment-gap objectives; {checks["small_positive_gap_objectives"]} small-positive-gap objectives. Degenerate settings remain in the geometry table; scores are undefined when the prescribed zero-gap rule applies.', '',
              f'Total measured worker time: {checks["worker_seconds"]/3600:.3f} CPU-hours, including preparation and output. This sum is not elapsed wall time; runtime_by_instance.csv and job accounting retain actual job/host evidence.', '',
              'Focused tests check source identities, record disjointness and nesting, normalization, rank invariance, tiny exact RAND enumeration, saved assignment reconstruction, endpoint aliases, deterministic scientific outcomes, checkpoint corruption, and parent-level aggregation. The pilot uses a separate phase/seed namespace, includes n=256,d=784 and a low-dimensional setting, and is excluded from production summaries.', '',
              '## Numerical record and reproduction','',
              '- [Canonical draw table](analysis_table.csv.gz)',
              '- [Geometry, ranks, scales, exact OPT and RAND](geometry_table.csv.gz)',
              '- [All setting means and parent intervals](summary.csv)',
              '- [Equal-dataset-weight means](equal_dataset_summary.csv)',
              '- [Instance noise summaries](instance_summaries.csv.gz)',
              '- [Parent summaries](parent_summaries.csv.gz)',
              '- [Paired mechanism differences](paired_comparisons.csv)',
              '- [Fixed-budget rank associations](rank_associations.csv)',
              '- [Panel aliases](panel_aliases.csv)',
              '- [Dictionary](data_dictionary.md)',
              '- [Frozen manifest](manifest.json)',
              '- [Verification counts and checksums](verification.json)',
              '- [Reproduction instructions](reproduce.md)', '',
              'The compressed numerical tables contain every setting and budget. The following small tables present all applicable mechanisms without displaying the draw-level record in this report.', '']
    lines.extend(f'- [{title}]({path})' for title,path in tables)
    lines += ['', '## Figure gallery','', 'Each PNG has a same-named PDF suitable for export.']
    for name, caption in links:
        lines += ['', f'- [{name}](figures/{name}.png) · [PDF](figures/{name}.pdf): {caption}']
    text='\n'.join(lines)+'\n'
    range_text = range_report(output)
    text = text.replace('## Aggregate comparisons\n', interpretation_text(summary)+'\n## Aggregate comparisons\n', 1)
    text = text.replace('## Bounds and geometric explanations\n',
                        range_text+'\n## Bounds and geometric explanations\n', 1)
    (output/'report.md').write_text(text)
    (output/'report.html').write_text(basic_html(text))


def reproduction(output, directory, manifest):
    text=f'''# Reproduction

Run only on a CRC compute node. The frozen package source and launcher are in
`{directory}/source_snapshot/`. The pinned source data are in
`{manifest['source_data']}`; they are archived beside the run after publication.
Dependency versions, pip freeze, Python version and NumPy build configuration
are saved in the run directory. Production pins one numerical thread.

Study fingerprint: `{manifest['fingerprint']}`. Scientific random seeds derive
from SHA256 of the saved namespace, phase, instance ID, draw and channel.
No process scheduling or shard identity enters a scientific random stream.
Pilot and production have separate streams. Keep source and numerical versions
identical: the worker rejects fingerprint drift. Reproduction on different
BLAS/hardware is not asserted bit-for-bit; verify the pinned environment and
compare saved outcome arrays and checksums. Runtime fields are measurements.

From the frozen environment on compute:

```bash
export PYTHONPATH={directory}/source_snapshot/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m private_matching study --study clustering_benchmark \\
  --source-data {manifest['source_data']} \\
  --results-root /path/to/new/results --stage prepare
python -m private_matching study --study clustering_benchmark \\
  --source-data {manifest['source_data']} \\
  --results-root /path/to/new/results --stage shard --shard-index 0 --shard-count 1
```

Use Slurm arrays with distinct shard indices for parallel reproduction. The
launcher `slurm/clustering_benchmark_submit.sh` creates a frozen source snapshot
and prepare/array/finalize dependencies. Resume with exactly the same directory
and source snapshot; committed files are verified before reuse. No two tasks
may own the same shard. Array concurrency changes elapsed time, never draws.

Regenerate tables and plots from existing checkpoints without simulation:

```bash
python -m private_matching study --study clustering_benchmark \\
  --results-root UNUSED --directory {directory} --stage finalize
```

Canonical export alone:

```bash
python -m private_matching analysis-table --input {directory} --output /path/to/handoff
```

Every selected original row ID and retained feature ID is in its geometry NPZ.
That file contains standardized Q/R, center and radius. Full source snapshots
permit exact raw reconstruction. JSON markers contain SHA256 of every committed
geometry/outcome file; saved permutations independently reconstruct every cost.
The canonical gzip table checksum verifies the exported record. Exact OPT
permutations and all realized private matchings are retained in outcome NPZs.

To check a fresh replay, compare `permutations`, `optimum_permutations`,
`total_cost`, `total_regret`, `row_oscillation`, and `realization_bound` arrays
for every instance, plus geometry arrays. Ignore runtime fields and archive
container timestamps. The focused validation suite includes an exact two-run
comparison of these scientific arrays.
'''
    (output/'reproduce.md').write_text(text)


def finalize(directory):
    directory=Path(directory)
    output=directory/'analysis_handoff'
    manifest,summary,cross,parent,expanded,checks=export(directory,output)
    links=plots(output,summary,cross,expanded,manifest)
    tables=readable_tables(output,summary,cross)
    reproduction(output,directory,manifest)
    report(output,manifest,summary,cross,checks,links,tables)
    atomic_json(directory/'COMPLETE.json',checks)
    print('BENCHMARK HANDOFF:',output,flush=True)
    return output


def geometry_range_comparisons(expanded):
    """Which observed geometry ranges favor which methods, with full coverage.

These are descriptive empirical conditional comparisons. Selection into bins is
not a proof for all possible Q/R in the bin. Keep the actual panel/label condition,
epsilon and objective; only aggregate spans distinct benchmark source corpora.
"""
    rank_edges = [0,1,2,4,8,16,32,64,128,256,512,1024]
    ratio_edges = [0,.125,.25,.5,.75,1.000000000001]
    rms_edges = [0,.125,.25,.5,.75,1.,1.5,2.]
    metrics = dict(q_rank=rank_edges, r_rank=rank_edges, q_effective_rank=rank_edges, r_effective_rank=rank_edges,
                   q_rank_ratio=ratio_edges, r_rank_ratio=ratio_edges, q_centered_rms=rms_edges,
                   r_centered_rms=rms_edges, side_center_separation=[0,.125,.25,.5,1.,2.,4.])
    rows = []
    context = ['panel','condition','objective_p','epsilon']
    for key, group in expanded.groupby(context, sort=True):
        common = dict(zip(context,key))
        for a,b in itertools.combinations(ARMS[common['objective_p']],2):
            left = group[group.mechanism==a].set_index('instance_id')
            right = group[group.mechanism==b].set_index('instance_id').reindex(left.index)
            if not left.index.equals(right.index):
                raise RuntimeError('geometry-bin instance pairing mismatch')
            diff = left.score-right.score
            # Channels are independent across arms except explicit endpoint reuse.
            # Across-epsilon coupling never supplies additional repetitions.
            source_a = left.source_arm.where(left.source_arm!='',a)
            source_b = right.source_arm.where(right.source_arm!='',b)
            same_channel = source_a == source_b
            noise_se = np.sqrt(left.noise_se_total_regret.astype(float)**2+right.noise_se_total_regret.astype(float)**2)/left.gap
            noise_se.loc[same_channel] = 0.
            for metric,edges in metrics.items():
                quantity = left[metric]
                for low,high in zip(edges[:-1],edges[1:]):
                    selected = (quantity>=low)&(quantity<high)&np.isfinite(diff)
                    if not selected.any(): continue
                    part = left.loc[selected]
                    differences = diff.loc[selected]
                    clusters = part[['dataset','parent']].copy()
                    clusters['difference'] = differences
                    # First partitions within dataset/parent; then datasets get
                    # equal weight within a product-parent block if present.
                    blocks = clusters.groupby(['dataset','parent']).difference.mean().reset_index()
                    parent_means = blocks.groupby('parent').difference.mean()
                    interval_result = interval(parent_means)
                    datasets = clusters.groupby('dataset').difference.mean()
                    rows.append(dict(common, mechanism_a=a, mechanism_b=b, diagnostic=metric,
                                     range_lower=low, range_upper=high,
                                     range_convention='lower inclusive, upper exclusive',
                                     instances=len(part), dataset_count=int(part.dataset.nunique()),
                                     dataset_instance_counts=json.dumps(part.dataset.value_counts().sort_index().to_dict(),sort_keys=True),
                                     observed_n=','.join(map(str,sorted(part.n.unique()))),
                                     observed_d=','.join(map(str,sorted(part.d.unique()))),
                                     observed_quantity_min=float(quantity[selected].min()), observed_quantity_max=float(quantity[selected].max()),
                                     mean_score_difference_instance_weighted=float(differences.mean()),
                                     mean_score_difference_equal_dataset_weight=float(datasets.mean()),
                                     minimum_observed_score_difference=float(differences.min()), maximum_observed_score_difference=float(differences.max()),
                                     fraction_observed_instance_means_favor_a=float(np.mean(differences < -1e-12)),
                                     fraction_observed_instance_means_favor_b=float(np.mean(differences > 1e-12)),
                                     fraction_tied=float(np.mean(np.abs(differences)<=1e-12)),
                                     all_observed_instance_means_favor_a=bool(np.all(differences < -1e-12)),
                                     all_observed_instance_means_favor_b=bool(np.all(differences > 1e-12)),
                                     noise_mean_SE_defined_instances=int(np.isfinite(noise_se[selected]).sum()),
                                     fraction_more_than_two_noise_SE_favor_a=float(np.mean((differences < -2*noise_se[selected])[np.isfinite(noise_se[selected])])) if np.isfinite(noise_se[selected]).any() else None,
                                     fraction_more_than_two_noise_SE_favor_b=float(np.mean((differences > 2*noise_se[selected])[np.isfinite(noise_se[selected])])) if np.isfinite(noise_se[selected]).any() else None,
                                     **{'parent_mean_difference_'+k:v for k,v in interval_result.items()},
                                     interpretation='exploratory observed conditional noise means; pointwise parent intervals; no simultaneous, universal or causal guarantee',
                                     private_diagnostic='NONPRIVATE_EXPLANATION_ONLY' if metric.startswith('r_') or metric=='side_center_separation' else 'public_geometry'))
    return rows


def range_report(output):
    frame = pd.read_csv(output/'geometry_range_comparisons.csv', dtype={'observed_n':str, 'observed_d':str}, low_memory=False)
    frame = frame[frame.epsilon==4.]
    lines = ['## Which geometry ranges favor which mechanisms?', '',
             'This analysis groups actual Q/R instances by fixed ranges of measured geometry. It asks whether a method wins on average and whether that advantage holds across all observed instance means in the group. Epsilon, objective, experiment panel and label condition remain separate. No bins were chosen from production winners.', '',
             'The compact example below compares Gumbel minus point-radial scores in the six-source aggregate panel at epsilon=4. A negative difference favors Gumbel. The win fraction counts the observed conditional means over the saved noise draws per arm, not a proof that the method wins on every possible Q/R or every draw. Instance counts and contributing corpora expose how much evidence lies in each range.', '',
             '| Objective | Public effective-rank range | Instances | Corpora | Equal-dataset mean difference | Observed means favoring Gumbel | All observed means favor Gumbel? |',
             '|---|---|---:|---:|---:|---:|---|']
    selected = frame[(frame.panel=='aggregate')&(frame.diagnostic=='q_effective_rank')&
                     (frame.mechanism_a=='gumbel')&(frame.mechanism_b=='ambient_radial')]
    for row in selected.sort_values(['objective_p','range_lower']).itertuples():
        lines.append(f'| {"Distance" if row.objective_p==1 else "Squared"} | [{row.range_lower:g}, {row.range_upper:g}) | {row.instances} | {row.dataset_count} | {row.mean_score_difference_equal_dataset_weight:.3f} | {100*row.fraction_observed_instance_means_favor_a:.1f}% | {"Yes" if row.all_observed_instance_means_favor_a else "No"} |')
    lines += ['', 'The full range table includes all applicable method pairs and both public/private ranks, effective ranks, rank ratios, side RMS radii and side-center separation. It retains extrema, tie fractions, parent-level differences and intervals, per-dataset contributions, and a descriptive two-noise-standard-error check. Overlapping datasets/conditions or many exploratory comparisons do not provide simultaneous guarantees.', '',
              '[All range-based comparisons](geometry_range_comparisons.csv)', '',
              'Small reading tables at epsilon=4 keep each panel and label condition separate:', '']
    for (panel,condition,p), group in frame.groupby(['panel','condition','objective_p']):
        pairs = [('gumbel','ambient_radial')]
        pairs += [('terminal_fold','anchor')] if p==1 else [('span_radial','coordinate_allocated')]
        name = f'geometry_ranges_{panel}_{condition}_p{p}.md'
        table = [f'# {panel}, {condition}, objective {p}, epsilon=4', '',
                 'Differences are first method minus second. Negative favors the first. Fractions describe observed instance noise means. Parent intervals and exact corpus contributions are in geometry_range_comparisons.csv.', '']
        for metric in ('q_effective_rank','r_effective_rank','q_rank_ratio','r_rank_ratio'):
            table += ['## '+metric.replace('_',' '), '',
                      '| Range | Comparison | Instances | Mean difference, equal corpus weights | First method wins | All observed means favor first? |',
                      '|---|---|---:|---:|---:|---|']
            for a,b in pairs:
                part = group[(group.diagnostic==metric)&(group.mechanism_a==a)&(group.mechanism_b==b)]
                for row in part.sort_values('range_lower').itertuples():
                    upper = min(row.range_upper,1.) if metric.endswith('ratio') else row.range_upper
                    table.append(f'| [{row.range_lower:g}, {upper:g}{"]" if upper==1. and metric.endswith("ratio") else ")"} | {LABELS[a]} minus {LABELS[b]} | {row.instances} | {row.mean_score_difference_equal_dataset_weight:.3f} | {100*row.fraction_observed_instance_means_favor_a:.1f}% | {"Yes" if row.all_observed_instance_means_favor_a else "No"} |')
            table.append('')
        (output/'tables'/name).write_text('\n'.join(table)+'\n')
        lines.append(f'- [{panel}, {condition}, objective {p}](tables/{name})')
    return '\n'.join(lines)+'\n'


def interpretation_text(summary):
    lines=['## What the comparisons show','',
           'Performance is compared at the same standardized inputs and epsilon. Gumbel releases only a trusted-curator matching; point, span, fold, anchor and cost-row mechanisms additionally protect reusable local reports. Utility comparisons therefore do not equate their operational capabilities.', '']
    for p in (1,2):
        objective='distance' if p==1 else 'squared cost'
        frame=summary[(summary.panel=='aggregate')&(summary.objective_p==p)&(summary.epsilon==4.)]
        winners=frame.loc[frame.groupby('dataset').score_mean.idxmin()]
        counts=winners.mechanism.value_counts()
        description=', '.join(f'{LABELS[arm]} on {count} of six corpora' for arm,count in counts.items())
        lines.append(f'For {objective} at epsilon=4, the lowest observed aggregate-dataset means are: {description}. Identical endpoint aliases can tie exactly, so a displayed winner label does not establish a strict advantage over its equivalent arm.')
        dim=summary[(summary.panel=='dimension')&(summary.objective_p==p)&(summary.epsilon==16.)]
        if len(dim):
            lines.append('')
            for arm in ('gumbel','ambient_radial'):
                part=dim[dim.mechanism==arm].set_index('d')
                lines.append(f'In the fixed n=256 MNIST dimension sweep at epsilon=16, {LABELS[arm]} has normalized regret {part.loc[6,"score_mean"]:.3f} at d=6 and {part.loc[784,"score_mean"]:.3f} at d=784; mean total regret is {part.loc[6,"mean_total_regret_mean"]:.3f} and {part.loc[784,"mean_total_regret_mean"]:.3f}, respectively.')
        sample=summary[(summary.panel=='sample_size')&(summary.objective_p==p)&(summary.epsilon==16.)&(summary.mechanism=='gumbel')].set_index('n')
        if len(sample):
            lines.append(f'At fixed d=784 and epsilon=16, Gumbel’s normalized regret is {sample.loc[16,"score_mean"]:.3f} at n=16 and {sample.loc[256,"score_mean"]:.3f} at n=256. Its total regret is {sample.loc[16,"mean_total_regret_mean"]:.3f} and {sample.loc[256,"mean_total_regret_mean"]:.3f}; normalized and absolute results answer different questions.')
        lines.append('')
    lines += ['Radial point noise acts in the ambient dimension; fold and span reports use dimensions determined by current public geometry. Coordinate allocation responds to public directional widths, while cost-row noise is calibrated to an entire destination row. Gumbel perturbs assignment scores directly. These construction differences motivate the recorded rank and geometry comparisons; the observed ranges below check whether the predicted advantages appear consistently.', '',
              'Coordinate deletion changes image information and geometry as well as dimension, even with pooled RMS fixed. The sweep therefore measures these sampled settings rather than isolating a universal dimension exponent or proving a threshold at d=n or d=log(n). Geometry-range comparisons help locate where each advantage holds and where exceptions remain.', '']
    return '\n'.join(lines)
