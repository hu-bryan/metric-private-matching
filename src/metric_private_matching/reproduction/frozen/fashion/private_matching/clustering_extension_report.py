"""Parallel CSV recording and separated, parent-level extension analyses."""
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
from scipy.spatial.distance import cdist

from . import clustering_benchmark as cb
from . import clustering_benchmark_report as br
from . import clustering_extension as ex
from .integral_report import COLORS,LABELS,basic_html
from .integral_mechanisms import Geometry
from .integral_study import atomic_json
from .login_guard import refuse_login_compute

VARIABLE_FIELDS=('objective_p','mechanism','epsilon','noise_draw','noise_coupling_id','total_cost','total_regret',
                 'normalized_score','optimum_total_cost','uniform_expected_total_cost','gap','gap_status','bound_total',
                 'bound_uncapped','bound_cap','bound_id','epsilon_y','epsilon_s','L_infinity','L_one',
                 'row_oscillation','realization_bound','runtime_seconds','solve_seconds','source_arm','reuse_reason',
                 'source_run_fingerprint','source_instance_id','privacy_output','original_units_rate_if_transform_frozen',
                 'raw_cost_multiplier','permutation_array','permutation_index')


def common_row(directory,meta,manifest):
    row=br.common_row(directory,meta,manifest)
    unit=meta['unit']
    row.update(study=ex.NAME,group=unit['group'],embedded=unit['embedded'],
               native_setting_id=unit['native_setting_id'],imported_original=bool(meta.get('imported_execution')),
               q_rho=row['q_rank_ratio'],r_rho=row['r_rank_ratio'],
               public_directional_width_certificate=meta['public'].get('coordinate_width_certificate'),
               public_diameter=meta['public'].get('public_diameter'),
               public_rank=meta['public'].get('public_rank'),fold_dimension=meta['public'].get('fold_dimension'))
    return row


def raw_fields():
    # Metadata fields are explicit and stable; this prototype also covers null
    # diagnostic cases without requiring a completed unit to create a header.
    unit=dict(instance_id='',setting_id='',dataset='',condition='',memberships=[],parent=0,partition=0,n=64,d=2,
              feature_order='native',group='',embedded=False,native_setting_id='')
    meta=dict(unit=unit,diagnostics={},degenerate=False,source_radius=1.,public={})
    return tuple(common_row(None,meta,dict(fingerprint='',design=dict(phase='production'))))+VARIABLE_FIELDS


def write_unit_csv(directory,manifest,meta,path):
    cfg=manifest['design'];unit=meta['unit'];n=unit['n'];fp=manifest['fingerprint']
    common=common_row(directory,meta,manifest)
    temp=Path(str(path)+'.tmp')
    with gzip.open(temp,'wt',newline='',compresslevel=3) as stream:
        writer=csv.writer(stream)
        if not meta['degenerate']:
            with np.load(directory/'units'/(unit['instance_id']+'.npz')) as saved:
                arrays={field:saved[field] for field in cb.FLOAT_FIELDS}
            for ip,p in enumerate((1,2)):
                opt=meta['optima'][str(p)];rand=meta['random_expected'][str(p)];gap=meta['gaps'][str(p)]
                for ia,arm in enumerate(cb.ARMS[p]):
                    reuse=meta['reuse'][f'{p}:{arm}']
                    if meta.get('imported_execution') or reuse['source_arm'].startswith('native:'):
                        origin=meta['execution_origin']
                        # Embedded Iris originates in this extension's native execution.
                        srcfp=origin['fingerprint'];srcid=origin['instance_id']
                        channel=arm
                        if reuse['source_arm'].startswith('native:'):
                            native=ex.native_unit(cfg,unit)
                            native_meta=json.loads((directory/'units'/(native['instance_id']+'.json')).read_text())
                            actual=native_meta['reuse'][f'{p}:{arm}']['source_arm']
                            channel=actual or arm
                        else: channel=reuse['source_arm'] or arm
                    else:
                        srcfp=fp;srcid=unit['instance_id'];channel=reuse['source_arm'] or arm
                    for ie,epsilon in enumerate(cfg['epsilon']):
                        cert=meta['bounds'][f'{p}:{arm}'][ie]
                        for draw in range(cfg['K']):
                            values={field:float(arrays[field][ip,ia,ie,draw]) for field in cb.FLOAT_FIELDS}
                            row=dict(objective_p=p,mechanism=arm,epsilon=epsilon,noise_draw=draw,
                                     noise_coupling_id=f'{srcfp}:{cfg["phase"]}:{srcid}:unit:{draw}:{channel}',
                                     **values,normalized_score=values['total_regret']/gap['value'] if gap['status']!='zero_gap' else None,
                                     optimum_total_cost=opt,uniform_expected_total_cost=rand,gap=gap['value'],gap_status=gap['status'],
                                     **cert,**reuse,source_run_fingerprint=srcfp,source_instance_id=srcid,
                                     privacy_output='final_matching_only' if arm=='gumbel' else 'local_report_and_matching',
                                     original_units_rate_if_transform_frozen=epsilon/meta['source_radius'],raw_cost_multiplier=meta['source_radius']**p,
                                     permutation_array='permutations',permutation_index=f'{ip},{ia},{ie},{draw}')
                            writer.writerow(list(common.values())+[row[name] for name in VARIABLE_FIELDS])
    os.replace(temp,path)


def validate(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['kind']!=ex.NAME:raise RuntimeError('wrong extension manifest')
    if not Path(manifest['source_data']).exists():
        manifest['source_data']=manifest['source_data'].replace('/vast/','/ix1/',1)
    ex.verify_runtime(manifest)
    expected=cb.all_units(manifest['design'])
    if len(list((directory/'units').glob('*.json')))!=len(expected): raise RuntimeError('STOP: extension unit completeness mismatch')
    for unit in expected:
        meta=cb.checkpoint(directory,unit,manifest['fingerprint'],verify=True)
        if meta is None or unit['instance_id']+'.csv.gz' not in meta['files']:raise RuntimeError('STOP: incomplete extension checkpoint')
    for name,record in manifest['sources']['files'].items():
        if cb.sha256(Path(manifest['source_data'])/name)!=record['sha256']:raise RuntimeError('STOP: source checksum mismatch')
    return manifest,expected


def export(directory,output):
    refuse_login_compute('export the clustering extension')
    directory,output=Path(directory),Path(output);manifest,expected=validate(directory)
    output.mkdir(parents=True,exist_ok=True);cfg=manifest['design']
    master=output/'analysis_table.csv.gz';temp=Path(str(master)+'.tmp')
    with gzip.open(temp,'wt',newline='') as handle:csv.writer(handle).writerow(raw_fields())
    instances=[];geometries=[];aliases=[];runtimes=[];embeddings=[];solves=0;rows=0;degenerate=0;imported=0
    with temp.open('ab') as merged:
        for unit in expected:
            stem=directory/'units'/unit['instance_id'];meta=json.loads(Path(str(stem)+'.json').read_text())
            with Path(str(stem)+'.csv.gz').open('rb') as source:shutil.copyfileobj(source,merged,1<<20)
            common=common_row(directory,meta,manifest)
            solves+=meta['actual_solves'];imported+=int(meta.get('imported_execution',False));degenerate+=int(meta['degenerate'])
            rows+=0 if meta['degenerate'] else 14*len(cfg['epsilon'])*cfg['K']
            runtimes.append(dict(instance_id=unit['instance_id'],elapsed_seconds=meta['elapsed_seconds'],
                                 process_peak_rss_kib=meta.get('process_peak_rss_kib'),actual_solves=meta['actual_solves'],imported_original=meta.get('imported_execution',False)))
            for membership in unit['memberships']:aliases.append(dict(instance_id=unit['instance_id'],**membership))
            if meta['embedding']:embeddings.append(dict(instance_id=unit['instance_id'],dataset=unit['dataset'],parent=unit['parent'],partition=unit['partition'],**meta['embedding']))
            if meta['degenerate']:
                geometries.append(dict(common,objective_p=None,optimum_total_cost=0.,random_total_cost=0.,gap=0.,gap_status='degenerate',status='degenerate'))
                continue
            with np.load(str(stem)+'.npz') as saved:
                costs=saved['total_cost'];regrets=saved['total_regret']
            for ip,p in enumerate((1,2)):
                gap=meta['gaps'][str(p)]
                geom=dict(common,objective_p=p,optimum_total_cost=meta['optima'][str(p)],random_total_cost=meta['random_expected'][str(p)],
                          gap=gap['value'],gap_status=gap['status'],gap_tolerance=gap['tolerance'],status='complete')
                geometries.append(geom)
                for ia,arm in enumerate(cb.ARMS[p]):
                    for ie,epsilon in enumerate(cfg['epsilon']):
                        value=regrets[ip,ia,ie];cert=meta['bounds'][f'{p}:{arm}'][ie]
                        instances.append(dict(geom,mechanism=arm,epsilon=epsilon,score=float(value.mean()/gap['value']) if gap['status']!='zero_gap' else None,
                                              mean_total_cost=float(costs[ip,ia,ie].mean()),mean_total_regret=float(value.mean()),
                                              noise_se_total_regret=float(value.std(ddof=1)/np.sqrt(cfg['K'])) if cfg['K']>1 else None,
                                              noise_sd_total_regret=float(value.std(ddof=1)) if cfg['K']>1 else None,
                                              bound_total=cert['bound_total'],K=cfg['K'],**meta['reuse'][f'{p}:{arm}']))
    os.replace(temp,master)
    br.write_csv(output/'geometry_table.csv.gz',geometries)
    br.write_csv(output/'instance_summaries.csv.gz',instances)
    br.write_csv(output/'panel_aliases.csv',aliases)
    br.write_csv(output/'runtime_by_instance.csv',runtimes)
    br.write_csv(output/'embedding_provenance.csv',embeddings)
    df=pd.DataFrame(instances)
    expanded=df.merge(pd.DataFrame(aliases)[['instance_id','panel']],on='instance_id',validate='many_to_many')
    keys=['panel','setting_id','dataset','condition','n','d','objective_p','mechanism','epsilon']
    metrics=['score','mean_total_regret','mean_total_cost','bound_total','q_rank','r_rank','q_effective_rank','r_effective_rank',
             'q_rank_ratio','r_rank_ratio','q_rho','r_rho','public_directional_width_certificate','public_diameter']
    parent=expanded.groupby(keys+['parent'])[metrics].mean().reset_index()
    parent['partition_count']=expanded.groupby(keys+['parent']).size().values
    parent.to_csv(output/'parent_summaries.csv.gz',index=False)
    summary=None
    for metric in metrics:
        part=br.summarize(parent,keys,metric)
        summary=part if summary is None else summary.merge(part,on=keys,validate='one_to_one')
    summary.to_csv(output/'summary.csv',index=False)
    group_frames=[]
    for panel in ('real','g2mg','h2mg'):
        blocks=parent[parent.panel==panel].groupby(['parent','objective_p','mechanism','epsilon'])[metrics].mean().reset_index()
        blocks['panel']=panel
        group_frames.append(br.summarize(blocks,['panel','objective_p','mechanism','epsilon'],'score'))
    group_summary=pd.concat(group_frames,ignore_index=True)
    group_summary.to_csv(output/'separate_group_summary.csv',index=False)
    paired,paired_parent=br.paired_comparisons(parent)
    br.write_csv(output/'paired_comparisons.csv',paired)
    br.write_csv(output/'paired_parent_differences.csv.gz',paired_parent)
    # Original range bins retained for rank/rho. Geometry groups remain separate.
    br.write_csv(output/'geometry_range_comparisons.csv',br.geometry_range_comparisons(expanded))
    associations=width_associations(parent)
    br.write_csv(output/'geometry_associations.csv',associations)
    control=control_comparisons(parent,manifest)
    br.write_csv(output/'embedding_paired_comparisons.csv',control)
    checks=dict(fingerprint=manifest['fingerprint'],instances=len(expected),new_simulated_instances=len(expected)-imported,
                imported_original_circles_instances=imported,canonical_draw_rows=rows,expected_draw_rows=(len(expected)-degenerate)*14*len(cfg['epsilon'])*cfg['K'],
                actual_new_assignment_solves=solves,degenerate_instances=degenerate,source_and_checkpoint_checksums_verified=True,
                embedding_instances=len(embeddings),all_embedding_distances_verified=all(e['all_pairwise_QR_verified'] for e in embeddings),
                canonical_sha256=cb.sha256(master),original_fingerprint=ex.ORIGINAL_FP,
                original_manifest_sha256=cb.sha256(ex.ORIGINAL/'manifest.json'),parent_collections=cfg['parents'],partitions_per_parent=cfg['partitions'],K=cfg['K'])
    if rows!=checks['expected_draw_rows']:raise RuntimeError('STOP: raw extension row mismatch')
    atomic_json(output/'verification.json',checks);atomic_json(output/'manifest.json',manifest)
    dictionary={name:br.field_description(name) for name in raw_fields()}
    dictionary.update(public_directional_width_certificate='W_Q=(sum public PCA coordinate widths^(2/3))^(3/2), existing coordinate-allocation certificate; public-only and in standardized distance units',
                      q_rho='effective rank / covariance rank, alias of q_rank_ratio',r_rho='effective rank / covariance rank, alias of r_rank_ratio; NONPRIVATE diagnostic',
                      source_run_fingerprint='Actual source execution fingerprint, including cross-instance isometry reuse and imported original Circles',
                      fold_dimension='Unchanged mechanism report dimension min(rank+1,d); embedding can add one fold noise coordinate')
    atomic_json(output/'data_dictionary.json',dictionary)
    (output/'data_dictionary.md').write_text(br.dictionary_text()+'\nThis separate extension has only epsilon sweeps at n=64. Real, g2mg, h2mg and embedding views remain separate. W_Q is the existing public PCA directional-width certificate. rho aliases effective-rank/rank. Original Circles draws are imported, not regenerated; invariant embedded arms reuse native samples with source fingerprints and instance IDs.\n')
    return manifest,summary,group_summary,parent,expanded,checks


def width_associations(parent):
    from scipy.stats import spearmanr
    result=[]
    for (panel,p),group in parent[parent.epsilon==4.].groupby(['panel','objective_p']):
        for a,b in itertools.combinations(cb.ARMS[p],2):
            left=group[group.mechanism==a].set_index(['setting_id','parent'])
            right=group[group.mechanism==b].set_index(['setting_id','parent']).reindex(left.index)
            difference=left.score-right.score
            for quantity in ('q_rank','r_rank','q_effective_rank','r_effective_rank','q_rho','r_rho','public_directional_width_certificate'):
                valid=np.isfinite(left[quantity])&np.isfinite(difference)
                rho=float(spearmanr(left.loc[valid,quantity],difference[valid]).statistic) if valid.sum()>=3 and left.loc[valid,quantity].nunique()>1 and difference[valid].nunique()>1 else None
                result.append(dict(panel=panel,objective_p=p,epsilon=4.,mechanism_a=a,mechanism_b=b,quantity=quantity,spearman_rho=rho,
                                   parent_setting_points=int(valid.sum()),interpretation='descriptive within group; correlated setting points share parents; no independent-point p value; not causal',
                                   private_diagnostic=quantity.startswith('r_')))
    return result


def control_comparisons(parent,manifest):
    rows=[]
    selected=parent[parent.panel=='embedding']
    for dataset in ('wut/circles','other/iris'):
        native='native_'+dataset.replace('/','_');embedded='embedded_'+dataset.replace('/','_')+'_d128'
        for p in (1,2):
            for arm in cb.ARMS[p]:
                for epsilon in manifest['design']['epsilon']:
                    left=selected[(selected.setting_id==embedded)&(selected.objective_p==p)&(selected.mechanism==arm)&(selected.epsilon==epsilon)].set_index('parent')
                    right=selected[(selected.setting_id==native)&(selected.objective_p==p)&(selected.mechanism==arm)&(selected.epsilon==epsilon)].set_index('parent')
                    delta=left.score-right.score
                    rows.append(dict(dataset=dataset,objective_p=p,mechanism=arm,epsilon=epsilon,
                                     difference='embedded minus native; positive means worse after isometry',
                                     **{'score_difference_'+k:v for k,v in br.interval(delta).items()},
                                     explicitly_reused_invariant_execution=arm in ex.INVARIANT[p]))
    return rows


def save_csv_table(frame,path,index='epsilon',p=1,title=''):
    lines=['# '+title,'','Mean normalized regret: 0 is OPT, 1 is exact RAND. Pointwise parent intervals are in summary.csv.','']
    lines+=['| '+index+' | '+' | '.join(LABELS[a] for a in cb.ARMS[p])+' |','|---|'+'---:|'*7]
    for value,group in frame.groupby(index,sort=True):
        by=group.set_index('mechanism')
        lines.append('| '+f'{value:g}'+' | '+' | '.join('undefined' if not np.isfinite(by.loc[a,'score_mean']) else f'{by.loc[a,"score_mean"]:.3f}' for a in cb.ARMS[p])+' |')
    path.write_text('\n'.join(lines)+'\n')


def plots_and_tables(output,summary,groups,manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    figures=output/'figures';tables=output/'tables';figures.mkdir(exist_ok=True);tables.mkdir(exist_ok=True)
    records=[]
    def save(fig,name,caption):
        fig.savefig(figures/(name+'.png'),dpi=150,bbox_inches='tight')
        fig.savefig(figures/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
        records.append(dict(name=name,caption=caption))
    def draw(ax,frame,p):
        for arm in cb.ARMS[p]:
            part=frame[frame.mechanism==arm].sort_values('epsilon')
            ax.plot(part.epsilon,part.score_mean,label=LABELS[arm],color=COLORS[arm],marker='o',markersize=3,lw=1.3)
            ax.fill_between(part.epsilon.to_numpy(float),part.score_low.to_numpy(float),part.score_high.to_numpy(float),color=COLORS[arm],alpha=.12)
        ax.set_xscale('log',base=2);ax.axhline(1,color='#555555',ls='--',lw=.7);ax.axhline(0,color='#999999',lw=.5)
        ticks=manifest['design']['epsilon'];ax.set_xticks(ticks,labels=[f'{e:g}' for e in ticks]);ax.tick_params(axis='x',labelsize=7)
        ax.set_xlabel('epsilon');ax.set_ylabel('Normalized regret');ax.grid(alpha=.15)
    for panel in ('real','g2mg','h2mg'):
        for p in (1,2):
            frame=summary[(summary.panel==panel)&(summary.objective_p==p)]
            count=frame.setting_id.nunique();cols=3;rows=(count+2)//3
            fig,axes=plt.subplots(rows,cols,figsize=(13,3.3*rows),squeeze=False)
            for ax,(sid,part) in zip(axes.flat,frame.groupby('setting_id',sort=True)):
                draw(ax,part,p);ax.set_title(part.dataset.iloc[0],fontsize=10)
                save_csv_table(part,tables/(sid+f'_p{p}.md'),p=p,title=part.dataset.iloc[0]+f', objective {p}, n=64 epsilon sweep')
            for ax in axes.flat[count:]:ax.axis('off')
            handles,labels=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=8)
            fig.suptitle(f'{panel}: {"Euclidean distance" if p==1 else "Squared Euclidean distance"}; n=64; joint pooled RMS=1')
            fig.tight_layout(rect=(0,.08,1,.96));save(fig,f'{panel}_epsilon_p{p}','Separate epsilon curves for every dataset, with pointwise 95% intervals over parent collections.')
    for p in (1,2):
        fig,axes=plt.subplots(1,3,figsize=(13,4.2))
        for ax,panel in zip(axes,('real','g2mg','h2mg')):
            draw(ax,groups[(groups.panel==panel)&(groups.objective_p==p)],p);ax.set_title(panel+' — equal dataset weights')
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=8)
        fig.tight_layout(rect=(0,.17,1,1));save(fig,f'separate_groups_p{p}','Each group gets its own summary; no overall average combines real and related synthetic datasets.')
        controls=pd.read_csv(output/'embedding_paired_comparisons.csv')
        fig,axes=plt.subplots(1,2,figsize=(11,4.8))
        for ax,dataset in zip(axes,('wut/circles','other/iris')):
            for arm in cb.ARMS[p]:
                part=controls[(controls.dataset==dataset)&(controls.objective_p==p)&(controls.mechanism==arm)].sort_values('epsilon')
                ax.plot(part.epsilon,part.score_difference_mean,color=COLORS[arm],label=LABELS[arm],marker='o',markersize=3)
                ax.fill_between(part.epsilon.to_numpy(float),part.score_difference_low.to_numpy(float),part.score_difference_high.to_numpy(float),color=COLORS[arm],alpha=.12)
            ax.set_title(dataset+' native → 128D isometry');ax.set_xscale('log',base=2);ax.axhline(0,color='#555555',ls='--',lw=.8)
            ax.set_xlabel('epsilon');ax.set_ylabel('Embedded minus native normalized regret');ax.grid(alpha=.15)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=8)
        fig.tight_layout(rect=(0,.17,1,1));save(fig,f'embedding_differences_p{p}','Paired parent differences. Positive means worse after distance-preserving embedding. Equivalent arms reuse native executions explicitly.')
        fig,axes=plt.subplots(2,2,figsize=(10,8))
        for row,dataset in enumerate(('wut/circles','other/iris')):
            for col,embedded in enumerate((False,True)):
                sid=('embedded_'+dataset.replace('/','_')+'_d128') if embedded else ('native_'+dataset.replace('/','_'))
                part=summary[(summary.panel=='embedding')&(summary.setting_id==sid)&(summary.objective_p==p)]
                draw(axes[row,col],part,p);axes[row,col].set_title(dataset+(' embedded 128D' if embedded else ' native'))
                save_csv_table(part,tables/(sid+f'_p{p}.md'),p=p,title=dataset+(' — embedded 128D' if embedded else ' — native')+f', objective {p}')
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=8)
        fig.tight_layout(rect=(0,.1,1,1));save(fig,f'embedding_curves_p{p}','Original Circles source selections are reused exactly; Iris controls use the new native Iris selections. No new n/d/label sweep.')
    # Width vs utility: parent means, within-group colors and no fitted causal line.
    frame=summary[(summary.epsilon==4.)&(summary.objective_p==2)&(summary.panel!='embedding')]
    left=frame[frame.mechanism=='coordinate_allocated'];right=frame[frame.mechanism=='span_radial']
    joined=left.merge(right,on=['panel','setting_id','dataset'],suffixes=('_allocated','_span'))
    fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    for ax,panel in zip(axes,('real','g2mg','h2mg')):
        part=joined[joined.panel==panel]
        ax.scatter(part.public_directional_width_certificate_mean_allocated,part.score_mean_allocated-part.score_mean_span,s=30)
        for row in part.itertuples():ax.annotate(row.dataset.split('/')[-1],(row.public_directional_width_certificate_mean_allocated,row.score_mean_allocated-row.score_mean_span),fontsize=6,xytext=(3,3),textcoords='offset points')
        ax.axhline(0,color='#555555',ls='--',lw=.8);ax.set_title(panel);ax.set_xlabel('Public directional-width certificate W_Q')
        ax.set_ylabel('Allocated minus span-radial score');ax.grid(alpha=.15)
    fig.suptitle('Squared cost, epsilon=4: descriptive geometry relationship')
    fig.tight_layout();save(fig,'directional_width_p2','One point per native dataset. Negative favors coordinate allocation. W_Q is a public certificate, not a guarantee of observed superiority; groups stay separate.')
    summary.to_csv(figures/'plot_inputs.csv',index=False)
    return records


def report(output,manifest,summary,groups,checks,figures):
    cfg=manifest['design'];phase=cfg['phase']
    def fmt(value):
        return 'undefined' if value is None or not np.isfinite(value) else f'{value:.3f}'
    text=['# Clustering benchmark extension results','',f'Completed {phase} extension `{manifest["fingerprint"]}`. Original results `{ex.ORIGINAL_FP}` remain unchanged.','',
          f'The record contains **{checks["instances"]:,} distinct instances**, including **{checks["new_simulated_instances"]:,} new simulated instances** and {checks["imported_original_circles_instances"]} read-only imported native Circles executions. Each production dataset uses 20 parent collections, five partitions and 50 noise draws per applicable arm and budget. All matching inputs have **n=64 public and 64 private points**.','',
          'Only epsilon varies within each dataset: 0.25, 0.5, 1, 2, 4, 8 and 16. The twelve synthetic datasets are existing benchmark files at dimensions 32/128 and spread parameters 20/50/80; each receives its own epsilon sweep. There are no new n, d or label-condition sweeps. Pilot draws use separate seeds and are excluded from production evidence.','',
          'Every actual Q/R instance is jointly centered and scaled to pooled RMS radius one before mechanisms run. All objectives and regrets are sums over the 64 matched pairs. Normalized regret is computed within each instance as mean regret divided by exact RAND minus exact OPT, before partition/parent averaging: 0 is OPT, 1 is exact random expectation, and values above 1 are retained.','',
          'The same nine mechanism labels and objective applicability are retained. Gumbel protects the final trusted-curator matching; the other arms protect reusable local reports. Privacy applies in standardized input coordinates, not to the data-dependent raw normalization or this public research archive.','',
          '## Real datasets','',
          'The nine real additions have a separate equal-dataset-weight summary. The table fixes epsilon=4 and shows descriptive lowest observed means, with pointwise 95% Student-t intervals over 20 parent means. Identical equivalent endpoints may tie.','',
          '| Dataset | Objective | Lowest observed mean | Score [95% interval] |','|---|---|---|---:|']
    for panel,title in [('real','Real datasets'),('g2mg','Gaussian family g2mg'),('h2mg','Hub family h2mg')]:
        if panel!='real':text+=['','## '+title,'','This family is summarized separately. No summary pools it with real data or the other synthetic family.','','| Dataset | Objective | Lowest observed mean | Score [95% interval] |','|---|---|---|---:|']
        for p in (1,2):
            for dataset,frame in summary[(summary.panel==panel)&(summary.objective_p==p)&(summary.epsilon==4.)].groupby('dataset'):
                best=frame.loc[frame.score_mean.idxmin()]
                text.append(f'| {dataset} | {"Distance" if p==1 else "Squared"} | {LABELS[best.mechanism]} | {fmt(best.score_mean)} [{fmt(best.score_low)}, {fmt(best.score_high)}] |')
        for p in (1,2):text+=['',f'![{panel} epsilon curves, objective {p}](figures/{panel}_epsilon_p{p}.png)']
    text+=['','## Isometric embedding controls','',
           'Circles (2D) and Iris (4D) are mapped to 128D by x → (x,0,…,0). The map matrix, source IDs, roles, ordering, normalization scales and full pooled-pairwise distance errors are saved. Native Circles reuses the completed benchmark selections and executions. Native Iris shares its single execution between the real-data and control views.','',
           'Both raw and standardized distances are checked for every control. Public ranks, PCA widths and basis alignment are checked before reusing invariant report laws. Gumbel and both cost-row mechanisms reuse native draws; anchor also reuses them for Euclidean cost, and span radial/coordinate allocation reuse them for squared cost. Their source execution IDs and noise channels are explicit in the canonical table.','',
           'Ambient point-noise laws still depend on ambient dimension. The Euclidean fold still uses min(rank+1,d): native full-span reports use 2/4 noise coordinates, whereas padded controls use 3/5. The extra private perpendicular-distance coordinate is not silently removed. Those arms receive their own draws and their comparisons are paired through parent collections.','']
    for p in (1,2):text+=[f'![Paired embedding difference, objective {p}](figures/embedding_differences_p{p}.png)','',f'[Native and embedded epsilon curves](figures/embedding_curves_p{p}.png)','']
    text+=['## Geometry and interpretation','',
           'rho is effective rank divided by covariance rank, an explicit alias of the existing rank ratio. The public directional-width certificate is W_Q = (sum of public PCA widths to power 2/3) to power 3/2. It is the certificate already used by coordinate allocation; the mechanism has not changed. Its squared-cost expected regret bound is 2n W_Q / epsilon. Rank, effective rank, rho and W_Q appear together in draw, geometry, instance, parent and setting summary tables.','',
           'The compact table below reports public geometry across parent collections. Small rho describes concentration of covariance energy relative to numerical rank. W_Q responds to public directional widths, while radial span noise is controlled by public span dimension and diameter. These are complementary attributes: numerical rank by itself does not encode anisotropy, and a smaller analytic width certificate does not prove smaller empirical regret.','',
           '| Group | Dataset | Public rank | Effective rank | rho | W_Q |','|---|---|---:|---:|---:|---:|']
    geom=summary[(summary.epsilon==4.)&(summary.objective_p==2)&(summary.mechanism=='coordinate_allocated')]
    for row in geom[geom.panel!='embedding'].sort_values(['panel','dataset']).itertuples():
        text.append(f'| {row.panel} | {row.dataset} | {row.q_rank_mean:.2f} | {row.q_effective_rank_mean:.2f} | {row.q_rho_mean:.3f} | {row.public_directional_width_certificate_mean:.3f} |')
    text+=['','![Directional width and allocation advantage](figures/directional_width_p2.png)','',
           'All applicable paired mechanism differences are retained. Rank/rho range comparisons keep the three groups and epsilon/objectives separate and show win fractions and exceptions. Width/rank associations at epsilon=4 are descriptive, preserve group identity, and have no independent-point p values or causal interpretation. Private-side attributes remain NONPRIVATE explanatory diagnostics and never tune the mechanisms.','',
           '## Record and validation','',
           f'Canonical private draw rows: **{checks["canonical_draw_rows"]:,}**. New exact assignment solves: **{checks["actual_new_assignment_solves"]:,}**. Degenerate zero-radius instances: {checks["degenerate_instances"]}. Every committed source/outcome checksum and every control distance certificate passed. Expected bounds are distinguished from checked realization certificates, including anchor secant checks.','',
           'Files are pinned to Clustering Benchmarks v1.2.0, immutable commit '+cb.COMMIT+'. Matrix, labels0, description and battery README files are retained with SHA256 and Git blob hashes. Data sampling is without replacement within each parent, public/private IDs are disjoint, and five partitions are averaged within parents before using the 20 parent means for intervals. Parent selections may reuse source rows across parents, conditional on the fixed corpus.','',
           '- [Canonical draw table](analysis_table.csv.gz)',
           '- [Geometry, ranks, rho and public widths](geometry_table.csv.gz)',
           '- [Setting means and pointwise parent intervals](summary.csv)',
           '- [Separate group summaries](separate_group_summary.csv)',
           '- [Paired mechanism comparisons](paired_comparisons.csv)',
           '- [Geometry-range comparisons](geometry_range_comparisons.csv)',
           '- [Geometry associations, including W_Q](geometry_associations.csv)',
           '- [Paired embedding comparisons](embedding_paired_comparisons.csv)',
           '- [Embedding provenance and distance errors](embedding_provenance.csv)',
           '- [Dictionary](data_dictionary.md)',
           '- [Manifest](manifest.json)',
           '- [Verification](verification.json)',
           '- [Reproduction instructions](reproduce.md)','',
           '## Readable epsilon tables','']
    for path in sorted((output/'tables').glob('*.md')):text.append(f'- [{path.stem}](tables/{path.name})')
    text+=['','## Figure downloads','']
    for fig in figures:text.append(f'- [{fig["name"]}](figures/{fig["name"]}.png) · [PDF](figures/{fig["name"]}.pdf): {fig["caption"]}')
    document='\n'.join(text)+'\n'
    document=document.replace('## Record and validation\n',empirical_interpretation(output,summary)+'\n## Record and validation\n',1)
    (output/'report.md').write_text(document);(output/'report.html').write_text(basic_html(document))


def reproduction(directory,output,manifest):
    (output/'reproduce.md').write_text(f'''# Reproduce the separate extension

Fingerprint: {manifest['fingerprint']}. Original benchmark {ex.ORIGINAL_FP} is read-only.
Hardware policy: AMD Turin, Slurm `--constraint=turin`, in every preparation,
simulation, regression, export and archived-environment replay stage. The frozen
environment check includes detected NumPy CPU dispatch and remains enabled.
Per-instance metadata and stage_records save actual host, CPU model and job IDs.
Production requests at most 125 concurrent single-core workers; QoS can permit fewer.
Sources are preserved at {manifest['source_data']}, also archived at the matching /ix1 path.
The complete frozen package, tests, scripts, dependency versions and NumPy build
configuration are saved under this run. The exact existing Python environment is
archived alongside the extension; its SHA256 is recorded. Only runtime/timestamps
are nondeterministic quantities. Scientific replay checks saved permutations and
costs within the documented floating-point tolerance.

Run only on CRC compute, one numerical thread. Launch commands are saved in
source_snapshot/slurm/clustering_extension_submit.sh. The array has one owner
for each native/control pair, uses separate pilot/production namespaces, and
validates every checkpoint SHA256 before reuse. No source or environment changes
are accepted by workers. Reproduction must retain the original Circles archive
or its copied native execution files plus provenance.

Regenerate tables and plots without simulation:

```bash
export PYTHONPATH={directory}/source_snapshot/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PM_EXT_CPU_CONSTRAINT=turin
python -m private_matching study --study clustering_extension --results-root UNUSED \\
  --directory {directory} --stage finalize
```

Canonical export alone uses `python -m private_matching analysis-table --input RUN --output OUTPUT`.
The full record stores standardized Q/R and all source row IDs, raw-source snapshots,
pooled normalization center/scale, embedding maps, public PCA geometry and spectra,
all private matchings, bound/sensitivity calibrations, explicit noise coupling and
cross-instance reuse, dataset/parent/partition identities and panel aliases.
Source-phase pilot records are never merged with production.
The immutable download helper is `source_snapshot/slurm/clustering_extension_fetch.sh`;
pass the destination, `source_snapshot/configs/clustering_extension_sources.sha256`
and `source_snapshot/configs/clustering_extension_pinned_git_tree.json`.
To rerun the entire suite on CRC, use the frozen launcher with arguments
`production 500 125 00:03:00 384M turin` after a successful independent pilot.
''')


def audit(directory,manifest):
    """Independent source reconstruction and cost audit across every setting."""
    cfg=manifest['design'];checked=0;instances=0
    for unit in cb.all_units(cfg):
        if (unit['parent'],unit['partition']) not in ((0,0),(cfg['parents']-1,cfg['partitions']-1)):continue
        stem=directory/'units'/unit['instance_id'];meta=json.loads(Path(str(stem)+'.json').read_text())
        prepared,_=ex.prepared_instance(manifest,unit);Q,R,records=prepared
        q,r,center,scale=cb.standardize(Q,R)
        if q is None:continue
        with np.load(str(stem)+'.geometry.npz') as saved,np.load(str(stem)+'.npz') as outcomes:
            assert np.allclose(saved['Q'],q,rtol=1e-12,atol=1e-12)
            assert np.allclose(saved['R'],r,rtol=1e-12,atol=1e-12)
            assert np.array_equal(saved['q_source_ids'],records['q_source_ids'])
            assert np.array_equal(saved['r_source_ids'],records['r_source_ids'])
            C=cdist(saved['R'],saved['Q'])
            for ip,p in enumerate((1,2)):
                pi=outcomes['permutations'][ip]
                assert np.all(np.sort(pi,axis=-1)==np.arange(64))
                true=(C**p)[np.arange(64),pi].sum(axis=-1)
                assert np.allclose(true,outcomes['total_cost'][ip],rtol=1e-12,atol=1e-10)
                checked+=true.size
            g=Geometry.build(saved['R'],saved['Q'])
            assert np.isclose(g.public_metadata()['coordinate_width_certificate'],meta['public']['coordinate_width_certificate'],rtol=1e-10,atol=1e-12)
        instances+=1
    evidence=dict(status='passed',independently_reconstructed_instances=instances,reconstructed_draw_labels=checked,
                  original_manifest_unchanged=cb.sha256(ex.ORIGINAL/'manifest.json')==manifest['original']['manifest_sha256'],
                  original_completion_unchanged=cb.sha256(ex.ORIGINAL/'COMPLETE.json')==manifest['original']['completion_sha256'])
    if not evidence['original_manifest_unchanged'] or not evidence['original_completion_unchanged']:raise RuntimeError('STOP: original benchmark changed')
    return evidence


def finalize(directory):
    directory=Path(directory);output=directory/'analysis_handoff'
    manifest,summary,groups,parent,expanded,checks=export(directory,output)
    aggregation=audit_aggregations(manifest,summary,groups,parent,expanded)
    atomic_json(output/'aggregation_verification.json',aggregation)
    figures=plots_and_tables(output,summary,groups,manifest)
    reproduction(directory,output,manifest)
    report(output,manifest,summary,groups,checks,figures)
    evidence=audit(directory,manifest);atomic_json(output/'audit_verification.json',evidence)
    atomic_json(directory/'COMPLETE.json',checks)
    print('EXTENSION HANDOFF:',output,flush=True)


def audit_aggregations(manifest,summary,groups,parent,expanded):
    """Check weighting and replication independently of the export's grouping."""
    from scipy.stats import t
    cfg=manifest['design']
    assert np.allclose(expanded.score,(expanded.mean_total_cost-expanded.optimum_total_cost)/expanded.gap,rtol=1e-11,atol=1e-11)
    assert (parent.partition_count==cfg['partitions']).all()
    keys=['panel','setting_id','dataset','condition','n','d','objective_p','mechanism','epsilon']
    expected=expanded.groupby(keys+['parent']).score.mean()
    actual=parent.set_index(keys+['parent']).score.reindex(expected.index)
    assert np.allclose(expected,actual,rtol=1e-12,atol=1e-12)
    for row in summary.itertuples():
        selector=np.ones(len(parent),dtype=bool)
        for key in keys:selector &= parent[key].to_numpy()==getattr(row,key)
        points=parent.loc[selector,'score'].to_numpy()
        assert len(points)==cfg['parents']
        assert np.isclose(points.mean(),row.score_mean,rtol=1e-12,atol=1e-12)
        if len(points)>1:
            half=t.ppf(.975,len(points)-1)*points.std(ddof=1)/np.sqrt(len(points))
            assert np.isclose(row.score_low,points.mean()-half,rtol=1e-12,atol=1e-12)
            assert np.isclose(row.score_high,points.mean()+half,rtol=1e-12,atol=1e-12)
    assert set(groups.panel)=={'real','g2mg','h2mg'}
    for row in groups.itertuples():
        frame=parent[(parent.panel==row.panel)&(parent.objective_p==row.objective_p)&
                     (parent.mechanism==row.mechanism)&(parent.epsilon==row.epsilon)]
        assert frame.setting_id.nunique()==(9 if row.panel=='real' else 6)
        points=frame.pivot(index='parent',columns='setting_id',values='score').mean(axis=1)
        assert len(points)==cfg['parents'] and np.isclose(points.mean(),row.score_mean,rtol=1e-12,atol=1e-12)
    return dict(status='passed',all_instance_scores_verified=True,partition_and_parent_weights_verified=True,
                all_setting_means_and_pointwise_intervals_verified=True,separate_equal_dataset_group_means_verified=True,
                embedding_controls_excluded_from_group_averages=True,parents=cfg['parents'],partitions=cfg['partitions'])


def empirical_interpretation(output,summary):
    lines=['## Observed mechanism advantages and geometric attributes','',
           'The following comparisons fix epsilon=4 and squared cost. Differences are coordinate allocation minus public-span radial, using the same parent collections. Negative favors allocation. The bound ratio is the ratio of the two mean analytic certificates, not an empirical utility prediction.','',
           '| Real dataset | Allocation minus span score [paired 95% interval] | Ratio of mean allocation/span bounds |',
           '|---|---:|---:|']
    pairs=pd.read_csv(output/'paired_comparisons.csv')
    chosen=pairs[(pairs.panel=='real')&(pairs.objective_p==2)&(pairs.epsilon==4.)&
                 (pairs.mechanism_a=='span_radial')&(pairs.mechanism_b=='coordinate_allocated')]
    lower=0
    for row in chosen.sort_values('dataset').itertuples():
        values=summary[(summary.panel=='real')&(summary.dataset==row.dataset)&(summary.objective_p==2)&(summary.epsilon==4.)].set_index('mechanism')
        ratio=values.loc['coordinate_allocated','bound_total_mean']/values.loc['span_radial','bound_total_mean']
        difference=-row.score_difference_mean
        lower+=int(difference<0)
        lines.append(f'| {row.dataset} | {difference:.3f} [{-row.score_difference_high:.3f}, {-row.score_difference_low:.3f}] | {ratio:.3f} |')
    lines+=['',f'Coordinate allocation has a lower observed mean on **{lower} of the nine real additions** in this fixed-budget comparison. Read its paired differences alongside rho and W_Q in the geometry table: numerical rank controls the size of an isotropic report, while anisotropy and directional widths affect how coordinate-specific rates spend the same privacy budget. The table retains disagreements between certificates and empirical means rather than using a bound as a winner selector.','',
            'The embedding controls provide a more direct check of ambient-dimension sensitivity: they hold row identities, true distances, OPT, RAND and intrinsic public geometry fixed. A change in an ambient arm therefore does not require a change of the matching problem. Invariant-arm differences near zero are explicit reuse checks, not independent replications.','',
            '| Control | Objective | Mechanism | Embedded minus native score [paired 95% interval], epsilon=4 |',
            '|---|---|---|---:|']
    controls=pd.read_csv(output/'embedding_paired_comparisons.csv')
    for p,arms in ((1,('ambient_radial','terminal_fold','gumbel')),(2,('ambient_radial','span_radial','coordinate_allocated'))):
        part=controls[(controls.objective_p==p)&(controls.epsilon==4.)&controls.mechanism.isin(arms)]
        for row in part.sort_values(['dataset','mechanism']).itertuples():
            lines.append(f'| {row.dataset} | {"Distance" if p==1 else "Squared"} | {LABELS[row.mechanism]} | {row.score_difference_mean:.3f} [{row.score_difference_low:.3f}, {row.score_difference_high:.3f}] |')
    lines+=['','The two synthetic families remain distinct in every summary. Their six settings each vary both source-file dimension and spread, so cross-setting changes in rank, rho or W_Q need not identify one isolated cause. The saved per-instance geometry and range tables support the narrower question of which observed Q/R ranges show consistent advantages and which contain exceptions.','']
    return '\n'.join(lines)
