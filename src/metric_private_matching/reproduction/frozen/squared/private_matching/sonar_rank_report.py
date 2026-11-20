"""Canonical Sonar handoff, conditional-noise summaries and readable figures."""
from __future__ import annotations
import csv
import gzip
import io
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr,t

from . import sonar_rank as sr
from . import clustering_benchmark as cb
from .integral_report import COLORS,LABELS,basic_html
from .integral_study import atomic_json
from .login_guard import refuse_login_compute


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    middle=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(middle-half),float(middle+half)


def distribution(values):
    x=np.asarray(values,float);n=len(x);mean=float(x.mean())
    se=float(x.std(ddof=1)/np.sqrt(n)) if n>1 else 0.
    half=float(t.ppf(.975,n-1))*se if n>1 else 0.
    return dict(mean=mean,mean_low=mean-half,mean_high=mean+half,sd=float(x.std(ddof=1)) if n>1 else 0.,
                q05=float(np.quantile(x,.05)),q25=float(np.quantile(x,.25)),median=float(np.median(x)),
                q75=float(np.quantile(x,.75)),q95=float(np.quantile(x,.95)),minimum=float(x.min()),maximum=float(x.max()))


def write_gzip_csv(path,rows,fields):
    # Exact canonical CSV bytes and gzip headers: runtime and file timestamps
    # never enter scientific tables. The same saved outputs give the same hash.
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=6) as zipped:
            with io.TextIOWrapper(zipped,encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
                for row in rows:writer.writerow({name:row.get(name) for name in fields})


def export(directory,output):
    refuse_login_compute('export the Sonar rank experiment')
    directory,output=Path(directory),Path(output)
    manifest=sr.relocation(json.loads((directory/'manifest.json').read_text()));sr.verify_runtime(manifest);sr.check_prepared(directory)
    cfg=manifest['design'];g=json.loads((directory/'geometry.json').read_text());output.mkdir(parents=True,exist_ok=True)
    rows=[];runtimes=[];all_decisions=[];observations=[];files=[]
    pcols=[f'p_bank_{bank+1:02d}' for bank in range(cfg['B'])]
    common=dict(study=sr.NAME,fingerprint=manifest['fingerprint'],phase=cfg['phase'],N=cfg['N'],d=cfg['d'],a=cfg['a'],b=cfg['b'],
                objective_p=2,energy_distance_p=1,public_rank=g['public_rank'],public_diameter=g['public_diameter'],
                public_directional_width_certificate=g['coordinate_width_certificate'],
                optimum_total_cost=g['optimum_total_cost'],random_expected_total_cost=g['random_expected_total_cost'],
                gap=g['gap']['value'],gap_status=g['gap']['status'],normalization='none; original signal units',
                J=cfg['J'],B=cfg['B'],primary_bank=1,alpha=cfg['alpha'])
    scalar=('total_cost','total_regret','normalized_regret','rank_T','swaps','W1_M','W1_R','Delta','statistic_change','realization_bound')
    for draw in range(cfg['K']):
        meta=sr.checkpoint(directory,draw,manifest['fingerprint'])
        if meta is None:raise RuntimeError('STOP: missing Sonar production repetition')
        runtimes.append(dict(draw=draw,elapsed_seconds=meta['elapsed_seconds'],peak_rss_kib=meta['peak_rss_kib'],**meta['hardware']))
        path=directory/'units'/f'draw{draw:04d}.npz';files.append(meta['files'])
        with np.load(path) as saved:
            values={key:saved[key] for key in scalar};pvalues=saved['rank_pvalues'];decisions=pvalues<=cfg['alpha']
            all_decisions.append(decisions);observations.append(values['rank_T'])
            for ia,arm in enumerate(sr.ARMS):
                for ie,epsilon in enumerate(cfg['epsilon']):
                    source=meta['aliases'].get(arm,arm);cert=g['bounds'][arm][ie]
                    record=dict(common,procedure='rank_energy',mechanism=arm,epsilon=epsilon,draw=draw,
                                noise_coupling_id=f'{manifest["fingerprint"]}:{cfg["phase"]}:draw{draw}:{source}',
                                source_arm=source if source!=arm else '',permutation_file=f'units/draw{draw:04d}.npz',
                                permutation_array='permutations',permutation_index=f'{ia},{ie}',
                                privacy_output='final_assignment_only' if arm=='gumbel' else 'reusable_local_reports',
                                postprocessing_of_same_ambient_report=ia<2,reference_reused_across_epsilon=False,
                                **{key:float(values[key][ia,ie]) for key in scalar},bound_total=cert['bound_total'],
                                primary_p=float(pvalues[ia,ie,0]),primary_reject=bool(decisions[ia,ie,0]),
                                p_bank_min=float(pvalues[ia,ie].min()),p_bank_max=float(pvalues[ia,ie].max()),
                                p_bank_sd=float(pvalues[ia,ie].std(ddof=1)),
                                decision_varies_across_banks=bool(np.any(decisions[ia,ie]!=decisions[ia,ie,0])),
                                **dict(zip(pcols,map(float,pvalues[ia,ie]))))
                    record['energy_E']=record['rank_T']*cfg['N']/(cfg['a']*cfg['b'])
                    record['normalized_swaps']=record['swaps']/min(cfg['a'],cfg['b'])
                    rows.append(record)
                    if ia<2:
                        p=float(saved['ordinary_pvalues'][ia,ie]);T=float(saved['ordinary_T'][ia,ie])
                        rows.append(dict(common,procedure='ordinary_energy',mechanism=arm,epsilon=epsilon,draw=draw,
                                         B=1,
                                         noise_coupling_id=record['noise_coupling_id'],source_arm=arm,
                                         privacy_output='postprocessing_of_same_reusable_ambient_report',postprocessing_of_same_ambient_report=True,
                                         reference_reused_across_epsilon=False,report_file=record['permutation_file'],
                                         report_array='ambient_reports',report_index=f'{ia},{ie}',rank_T=T,
                                         energy_E=T*cfg['N']/(cfg['a']*cfg['b']),primary_p=p,primary_reject=p<=cfg['alpha'],
                                         p_bank_01=p))
            p=saved['random_pvalues'];T=float(saved['random_T']);cost=float(saved['random_cost'])
            rows.append(dict(common,procedure='rank_energy',mechanism='uniform_random',epsilon=None,draw=draw,
                             noise_coupling_id=f'{manifest["fingerprint"]}:uniform-rank-reference:{draw}',source_arm='',
                             privacy_output='input_independent',reference_reused_across_epsilon=True,
                             total_cost=cost,total_regret=float(saved['random_regret']),normalized_regret=float(saved['random_regret'])/g['gap']['value'],
                             rank_T=T,energy_E=T*cfg['N']/(cfg['a']*cfg['b']),swaps=float(saved['random_swaps']),
                             normalized_swaps=float(saved['random_swaps'])/cfg['b'],Delta=float(saved['random_Delta']),
                             statistic_change=abs(T-g['nonprivate_T']),primary_p=float(p[0]),primary_reject=bool(p[0]<=cfg['alpha']),
                             p_bank_min=float(p.min()),p_bank_max=float(p.max()),p_bank_sd=float(p.std(ddof=1)),
                             decision_varies_across_banks=bool(np.any((p<=cfg['alpha'])!=(p[0]<=cfg['alpha']))),
                             permutation_file=f'units/draw{draw:04d}.npz',permutation_array='random_permutation',permutation_index='',
                             **dict(zip(pcols,map(float,p)))))
    with np.load(directory/'banks.npz') as banks:
        nonprivate_p=banks['nonprivate_pvalues'];bankT=banks['statistics']
        boundary=float(np.sort(bankT[0])[int(np.ceil((1-cfg['alpha'])*(cfg['J']+1)))-1])
    rows.append(dict(common,procedure='rank_energy',mechanism='nonprivate',epsilon=None,draw=None,
                     privacy_output='NONPRIVATE_REFERENCE',reference_reused_across_epsilon=True,
                     total_cost=g['optimum_total_cost'],total_regret=0.,normalized_regret=0.,rank_T=g['nonprivate_T'],energy_E=g['nonprivate_E'],
                     swaps=0.,normalized_swaps=0.,Delta=0.,statistic_change=0.,primary_p=float(nonprivate_p[0]),
                     primary_reject=bool(nonprivate_p[0]<=cfg['alpha']),p_bank_min=float(nonprivate_p.min()),p_bank_max=float(nonprivate_p.max()),
                     p_bank_sd=float(nonprivate_p.std(ddof=1)),decision_varies_across_banks=bool(np.any((nonprivate_p<=cfg['alpha'])!=(nonprivate_p[0]<=cfg['alpha']))),
                     permutation_file='inputs.npz',permutation_array='nonprivate_permutation',permutation_index='',
                     **dict(zip(pcols,map(float,nonprivate_p)))))
    fields=list(dict.fromkeys(key for row in rows for key in row))
    write_gzip_csv(output/'analysis_table.csv.gz',rows,fields)
    df=pd.DataFrame(rows);summary=[];bank_rows=[];associations=[];paired=[]
    metrics=('rank_T','primary_p','total_cost','total_regret','normalized_regret','swaps','normalized_swaps','Delta','statistic_change')
    for (procedure,arm,epsilon),part in df.groupby(['procedure','mechanism','epsilon'],dropna=False,sort=True):
        n=len(part);rejections=int(part.primary_reject.sum());lo,hi=wilson(rejections,n)
        if arm=='nonprivate':lo=hi=None
        rec=dict(procedure=procedure,mechanism=arm,epsilon=None if pd.isna(epsilon) else float(epsilon),K=n,
                 rejection_count=rejections,rejection_frequency=rejections/n,rejection_low=lo,rejection_high=hi)
        for metric in metrics:
            values=part[metric].dropna()
            if len(values):rec.update({metric+'_'+key:value for key,value in distribution(values).items()})
        if procedure=='rank_energy':
            counts=[]
            for ib,column in enumerate(pcols):
                k=int((part[column]<=cfg['alpha']).sum());lower,upper=wilson(k,n);counts.append(k/n)
                if arm=='nonprivate':lower=upper=None
                bank_rows.append(dict(procedure=procedure,mechanism=arm,epsilon=rec['epsilon'],bank=ib+1,K=n,rejection_count=k,
                                      rejection_frequency=k/n,rejection_low=lower,rejection_high=upper))
            rec.update(bank_frequency_min=min(counts),bank_frequency_max=max(counts),bank_frequency_sd=float(np.std(counts,ddof=1)),
                       fixed_output_decision_disagreement_fraction=float(part.decision_varies_across_banks.mean()))
            if arm in sr.ARMS:
                for x,y in (('normalized_regret','rank_T'),('normalized_regret','Delta'),('normalized_regret','swaps'),('Delta','statistic_change')):
                    correlation=float(spearmanr(part[x],part[y]).statistic) if part[x].nunique()>1 and part[y].nunique()>1 else None
                    associations.append(dict(mechanism=arm,epsilon=epsilon,x=x,y=y,spearman_rho=correlation,
                                             interpretation='descriptive within fixed method/budget; no cross-epsilon independence or causal claim'))
        summary.append(rec)
    summary=pd.DataFrame(summary);summary.to_csv(output/'summary.csv',index=False,float_format='%.17g')
    pd.DataFrame(bank_rows).to_csv(output/'bank_rejection_frequencies.csv',index=False,float_format='%.17g')
    pd.DataFrame(associations).to_csv(output/'diagnostic_associations.csv',index=False,float_format='%.17g')
    for arm in sr.ARMS[:2]:
        for epsilon in cfg['epsilon']:
            rank=df[(df.procedure=='rank_energy')&(df.mechanism==arm)&(df.epsilon==epsilon)].set_index('draw').sort_index()
            ordinary=df[(df.procedure=='ordinary_energy')&(df.mechanism==arm)&(df.epsilon==epsilon)].set_index('draw').sort_index()
            delta=rank.primary_reject.astype(int)-ordinary.primary_reject.astype(int)
            paired.append(dict(mechanism=arm,epsilon=epsilon,rank_rejection_frequency=float(rank.primary_reject.mean()),
                               ordinary_rejection_frequency=float(ordinary.primary_reject.mean()),
                               rank_only_rejections=int((rank.primary_reject&~ordinary.primary_reject).sum()),
                               ordinary_only_rejections=int((~rank.primary_reject&ordinary.primary_reject).sum()),
                               paired_difference=float(delta.mean()),**{'paired_difference_'+k:v for k,v in distribution(delta).items()}))
    pd.DataFrame(paired).to_csv(output/'ambient_report_comparison.csv',index=False,float_format='%.17g')
    with np.load(directory/'null_check.npz') as null:
        null_p=null['pvalues'];null_rows=[]
        for ib in range(cfg['B']):
            k=int(np.count_nonzero(null_p[:,ib]<=cfg['alpha']));lower,upper=wilson(k,cfg['H'])
            null_rows.append(dict(bank=ib+1,H=cfg['H'],rejections=k,rejection_frequency=k/cfg['H'],low=lower,high=upper,
                                  interpretation='same independent null cases across banks; not method-specific or pooled B*H trials'))
        write_gzip_csv(output/'null_cases.csv.gz',[
            dict(case=i,T=float(null['statistics'][i]),**dict(zip(pcols,map(float,null_p[i])))) for i in range(cfg['H'])],['case','T']+pcols)
    pd.DataFrame(null_rows).to_csv(output/'null_calibration.csv',index=False,float_format='%.17g')
    near=[]
    for arm in sr.ARMS:
        candidates=df[(df.procedure=='rank_energy')&(df.mechanism==arm)&(df.epsilon==16.)].copy()
        candidates['distance_to_threshold']=abs(candidates.primary_p-.05)
        example=candidates.sort_values(['distance_to_threshold','draw'],kind='stable').iloc[0]
        for ib,column in enumerate(pcols):near.append(dict(mechanism=arm,epsilon=16.,draw=int(example.draw),rank_T=float(example.rank_T),
                                                        bank=ib+1,pvalue=float(example[column]),reject=bool(example[column]<=.05),
                                                        selection='one fixed output per arm at epsilon16 closest primary p to .05, ties by draw; descriptive illustration only'))
    pd.DataFrame(near).to_csv(output/'near_threshold_examples.csv',index=False,float_format='%.17g')
    pd.DataFrame(runtimes).to_csv(output/'runtime_by_draw.csv',index=False)
    checks=dict(status='passed',fingerprint=manifest['fingerprint'],phase=cfg['phase'],private_draw_rows=cfg['K']*7*7,
                ambient_ordinary_rows=cfg['K']*2*7,uniform_reference_rows=cfg['K'],nonprivate_rows=1,
                canonical_rows=len(df),public_banks=cfg['B'],allocations_per_bank=cfg['J'],independent_null_cases=cfg['H'],
                primary_bank=1,nonprivate_T=g['nonprivate_T'],nonprivate_primary_p=float(nonprivate_p[0]),
                nonprivate_reject=bool(nonprivate_p[0]<=cfg['alpha']),nonprivate_p_min=float(nonprivate_p.min()),nonprivate_p_max=float(nonprivate_p.max()),
                grid_affine_rank=g['public_rank'],primary_T_boundary_strictly_greater=boundary,
                sources_preparation_and_all_unit_checksums_verified=True,canonical_sha256=cb.sha256(output/'analysis_table.csv.gz'),
                pilot_production_pooling=False,primary_intervals='pointwise Wilson over independent private repetitions, conditional on bank1')
    if len(df)!=cfg['K']*64+1:raise RuntimeError('STOP: incomplete Sonar canonical export')
    atomic_json(output/'verification.json',checks);atomic_json(output/'manifest.json',manifest);atomic_json(output/'geometry.json',g)
    dictionary=dict(rank_T='T=ab/N times biased energy using UNSQUARED Euclidean distances, including zero diagonals',
                    primary_reject='plus-one Monte Carlo p <= .05, primary bank1 fixed in advance',
                    total_regret='NONPRIVATE experimenter diagnostic: sum true SQUARED matching cost minus exact optimum',
                    normalized_regret='total regret / (exact uniform expected total cost - exact optimum); values >1 retained',
                    swaps='minimum cross-cohort swaps, with within-cohort rearrangements free',Delta='exact W1_M + W1_R on public grid distances',
                    noise_coupling_id='same unit private noise across epsilon; method channels independent except explicit alias',
                    ordinary_energy='postprocessing of saved ambient_reports; report-dependent permutation statistics recomputed',
                    bank_sensitivity='20 assessments of each fixed private output; never 20*K independent private trials',
                    source_files='inputs.npz, banks.npz, null_check.npz and units/drawNNNN.npz in run archive; SHA256 checked')
    atomic_json(output/'data_dictionary.json',dictionary)
    return manifest,df,summary,checks


def plot_all(output,manifest,df,summary,checks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.compression':6})
    figdir=output/'figures';figdir.mkdir(exist_ok=True);figures=[];eps=manifest['design']['epsilon']
    rank=summary[(summary.procedure=='rank_energy')&summary.mechanism.isin(sr.ARMS)]
    refs=summary[(summary.procedure=='rank_energy')&~summary.mechanism.isin(sr.ARMS)].set_index('mechanism')
    def save(fig,name,caption):
        fig.savefig(figdir/(name+'.png'),dpi=160,bbox_inches='tight',metadata={'Software':'private_matching Sonar fixed-profile export'})
        fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
        figures.append(dict(name=name,caption=caption))
    def setup(ax,ylabel):
        ax.set_xscale('log',base=2);ax.set_xticks(eps,labels=[f'{e:g}' for e in eps]);ax.set_xlabel('epsilon (original signal units)');ax.set_ylabel(ylabel);ax.grid(alpha=.17)
    def lines(ax,frame,metric,low=None,high=None):
        for arm in sr.ARMS:
            part=frame[frame.mechanism==arm].sort_values('epsilon')
            ax.plot(part.epsilon,part[metric],color=COLORS[arm],label=LABELS[arm],marker='o',ms=3)
            if low:ax.fill_between(part.epsilon.to_numpy(float),part[low].to_numpy(float),part[high].to_numpy(float),color=COLORS[arm],alpha=.13)
    def legend(fig,ax):
        handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=8)
    fig,ax=plt.subplots(figsize=(11,5.8));lines(ax,rank,'rejection_frequency','rejection_low','rejection_high')
    ax.axhline(float(refs.loc['nonprivate','rejection_frequency']),color='black',ls='--',label='Nonprivate fixed decision')
    ax.axhline(float(refs.loc['uniform_random','rejection_frequency']),color='#777777',ls=':',label='Shared uniform-rank reference')
    ax.set_ylim(-.02,1.02);setup(ax,'Conditional rejection frequency over privacy noise');legend(fig,ax)
    fig.tight_layout(rect=(0,.16,1,1));save(fig,'rank_rejection','Primary bank1; pointwise 95% Wilson intervals over K independent privacy draws. References are shared across epsilon; this is not population power.')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,arm in zip(axes,sr.ARMS[:2]):
        for procedure,style in (('rank_energy','-'),('ordinary_energy','--')):
            part=summary[(summary.mechanism==arm)&(summary.procedure==procedure)].sort_values('epsilon')
            ax.plot(part.epsilon,part.rejection_frequency,style,marker='o',label=procedure.replace('_',' '))
            ax.fill_between(part.epsilon.to_numpy(float),part.rejection_low.to_numpy(float),part.rejection_high.to_numpy(float),alpha=.13)
        comparison=summary[summary.mechanism==arm]
        ax.axhline(.05,color='#777777',ls=':');ax.set_ylim(0,min(1.,max(.15,float(comparison.rejection_high.max())*1.15)))
        ax.set_title(LABELS[arm]);setup(ax,'Conditional rejection frequency');ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'ordinary_report_comparison','Same ambient reports serve both tests. Ordinary-energy null statistics are recomputed separately for each report with 9999 allocations; no Q-grid null substitution.')
    for metric,name,ylabel,reference in (('rank_T','rank_energy_distributions','Rank-energy statistic T',checks['nonprivate_T']),
                                        ('primary_p','rank_pvalue_distributions','Primary Monte Carlo p-value',checks['nonprivate_primary_p'])):
        fig,axes=plt.subplots(2,4,figsize=(15,7.5))
        for ax,arm in zip(axes.flat,sr.ARMS):
            part=rank[rank.mechanism==arm].sort_values('epsilon')
            ax.fill_between(part.epsilon.to_numpy(float),part[metric+'_q05'].to_numpy(float),part[metric+'_q95'].to_numpy(float),color=COLORS[arm],alpha=.16,label='Individual outputs: 5–95%')
            ax.fill_between(part.epsilon.to_numpy(float),part[metric+'_q25'].to_numpy(float),part[metric+'_q75'].to_numpy(float),color=COLORS[arm],alpha=.27,label='Individual outputs: 25–75%')
            ax.plot(part.epsilon,part[metric+'_median'],color=COLORS[arm],marker='o',label='Median')
            ax.axhline(reference,color='black',ls='--',lw=1,label='Nonprivate value')
            ax.axhline(.05 if metric=='primary_p' else checks['primary_T_boundary_strictly_greater'],color='#bb3333',ls=':',label='Decision boundary')
            ax.set_title(LABELS[arm],fontsize=10);setup(ax,ylabel)
            if metric=='primary_p':ax.set_ylim(0,1.02)
        axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,fontsize=8);fig.tight_layout(rect=(0,.08,1,1))
        save(fig,name,'Bands show distributions of individual private outputs, not confidence intervals for means. Unsquared grid distances define T; matching uses squared cost.')
    fig,axes=plt.subplots(1,2,figsize=(13,5.5));lines(axes[0],rank,'rejection_frequency','bank_frequency_min','bank_frequency_max')
    lines(axes[1],rank,'fixed_output_decision_disagreement_fraction')
    setup(axes[0],'Rejection frequency: primary line, 20-bank range');setup(axes[1],'Fraction of fixed outputs changing decision')
    for ax in axes:ax.set_ylim(0,max(.12,float(rank.bank_frequency_max.max())*1.15))
    legend(fig,axes[0]);fig.tight_layout(rect=(0,.17,1,1));save(fig,'bank_robustness','Bank ranges measure permutation Monte Carlo sensitivity. They are not confidence intervals and do not add private replications.')
    null=pd.read_csv(output/'null_calibration.csv');fig,ax=plt.subplots(figsize=(10,4))
    ax.errorbar(null.bank,null.rejection_frequency,yerr=np.array([null.rejection_frequency-null.low,null.high-null.rejection_frequency]),fmt='o',capsize=2)
    ax.axhline(.05,color='black',ls='--');ax.set_xlabel('Independent public calibration bank');ax.set_ylabel('Null rejection frequency');ax.set_xticks(null.bank);ax.set_ylim(0,max(.12,float(null.high.max())*1.05));ax.grid(alpha=.17)
    fig.tight_layout();save(fig,'null_calibration','H fresh independent public null labelings are assessed against every bank; Wilson intervals are over H cases conditional on a bank. The same H cases are reused across banks.')
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for ax,metric,label in zip(axes,('total_regret','normalized_regret'),('Mean total squared regret','Mean regret / (RAND − OPT)')):
        lines(ax,rank,metric+'_mean',metric+'_mean_low',metric+'_mean_high');setup(ax,label)
        ax.axhline(float(refs.loc['uniform_random',metric+'_mean']),color='#777777',ls=':',label='Shared random reference')
    axes[1].axhline(1.,color='black',ls='--',lw=.8);legend(fig,axes[0]);fig.tight_layout(rect=(0,.18,1,1))
    save(fig,'squared_regret','True raw-data squared costs are NONPRIVATE experimenter diagnostics. Bands are pointwise 95% intervals for means over privacy noise; all totals sum 208 pairs.')
    fig,axes=plt.subplots(1,3,figsize=(15,5.5))
    for ax,metric,label in zip(axes,('swaps','normalized_swaps','Delta'),('Mean cross-cohort swaps s','Mean normalized swaps s/97','Mean cohort Wasserstein sum Delta')):
        lines(ax,rank,metric+'_mean',metric+'_mean_low',metric+'_mean_high');setup(ax,label)
        ax.axhline(float(refs.loc['uniform_random',metric+'_mean']),color='#777777',ls=':')
    legend(fig,axes[0]);fig.tight_layout(rect=(0,.18,1,1));save(fig,'cohort_discrepancy','Exact minimum cross-cohort exchanges and exact metric W1 discrepancy ignore within-cohort rearrangements. Bands quantify uncertainty of conditional means, not individual-output spreads.')
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for arm in sr.ARMS:
        part=rank[rank.mechanism==arm].sort_values('epsilon')
        for ax,x,y in ((axes[0],'Delta_mean','statistic_change_mean'),(axes[1],'normalized_regret_mean','rejection_frequency')):
            ax.plot(part[x],part[y],color=COLORS[arm],marker='o',ms=4,label=LABELS[arm],lw=.8)
            final=part.iloc[-1];ax.annotate('16',(final[x],final[y]),fontsize=7)
    axes[0].set_xlabel('Mean geometric cohort discrepancy Delta');axes[0].set_ylabel('Mean |T − T(nonprivate)|')
    axes[1].set_xlabel('Mean normalized squared regret');axes[1].set_ylabel('Primary conditional rejection frequency')
    for ax in axes:ax.grid(alpha=.17)
    legend(fig,axes[0]);fig.tight_layout(rect=(0,.2,1,1));save(fig,'diagnostic_relationships','One point per arm/budget; connected points share unit privacy noise across epsilon. Labels identify epsilon16. Descriptive relationships do not supply independent cross-budget evidence.')
    near=pd.read_csv(output/'near_threshold_examples.csv');fig,ax=plt.subplots(figsize=(11,5.5))
    for arm in sr.ARMS:
        part=near[near.mechanism==arm];ax.plot(part.bank,part.pvalue,color=COLORS[arm],marker='o',ms=3,lw=1,label=LABELS[arm])
    ax.axhline(.05,color='black',ls='--');ax.set_xlabel('Public calibration bank');ax.set_ylabel('p-value of the SAME fixed private output');ax.grid(alpha=.17)
    legend(fig,ax);fig.tight_layout(rect=(0,.2,1,1));save(fig,'fixed_output_bank_pvalues','One epsilon16 output per arm, selected only for illustration by closest primary p to .05. Assignment and T stay fixed; only the calibration bank changes.')
    return figures


def audit(directory,manifest):
    cfg=manifest['design'];ctx=sr.load_context(directory,manifest);data=ctx['inputs'];g=ctx['geometry'];geo=ctx['geo'];checked=0
    metal=data['metal'];D=data['grid_distances'];n=cfg['N'];a=cfg['a'];b=cfg['b']
    with np.load(directory/'banks.npz') as banks:
        assert np.all(banks['allocations'].sum(axis=2)==a)
        for ib in range(cfg['B']):
            indices=[0,cfg['J']//2,cfg['J']-1];masks=banks['allocations'][ib,indices]
            literal=[]
            for mask in masks:
                E=2*D[mask][:,~mask].sum()/(a*b)-D[mask][:,mask].sum()/a**2-D[~mask][:,~mask].sum()/b**2
                literal.append(a*b/n*E)
            assert np.allclose(literal,banks['statistics'][ib,indices],rtol=1e-12,atol=1e-12)
        ordered=banks['sorted_statistics']
    for draw in sorted({0,cfg['K']-1}):
        with np.load(directory/'units'/f'draw{draw:04d}.npz') as saved:
            pi=saved['permutations'];assert np.all(np.sort(pi,axis=-1)==np.arange(n))
            costs=geo.C2[np.arange(n),pi].sum(axis=-1)
            assert np.array_equal(costs,saved['total_cost'])
            assert np.allclose(costs-g['optimum_total_cost'],saved['total_regret'],rtol=1e-12,atol=1e-12)
            for ia,arm in enumerate(sr.ARMS):
                for ie,epsilon in enumerate(cfg['epsilon']):
                    chosen=pi[ia,ie];mask=sr.slot_membership(chosen,metal)
                    E=2*D[mask][:,~mask].sum()/(a*b)-D[mask][:,mask].sum()/a**2-D[~mask][:,~mask].sum()/b**2
                    T=a*b/n*E
                    assert np.isclose(T,saved['rank_T'][ia,ie],rtol=1e-12,atol=1e-12)
                    assert np.array_equal(sr.bank_pvalues(saved['rank_T'][ia,ie],ordered),saved['rank_pvalues'][ia,ie])
                    s,wM,wR=sr.cohort_discrepancy(chosen,data['nonprivate_permutation'],metal,D)
                    assert s==saved['swaps'][ia,ie] and np.isclose(wM+wR,saved['Delta'][ia,ie],atol=1e-12)
                    if ia<2:
                        report=data['R']+saved['unit_noise_'+arm]/epsilon
                        assert np.array_equal(report,saved['ambient_reports'][ia,ie])
                        distance=cdist(report,report)
                        ordinary=a*b/n*(2*distance[metal][:,~metal].sum()/(a*b)-distance[metal][:,metal].sum()/a**2-distance[~metal][:,~metal].sum()/b**2)
                        assert np.isclose(ordinary,saved['ordinary_T'][ia,ie],rtol=1e-11,atol=1e-10)
                        direct=(1+np.count_nonzero(saved['ordinary_null_statistics'][ia,ie]>=saved['ordinary_T'][ia,ie]))/(cfg['J']+1)
                        assert direct==saved['ordinary_pvalues'][ia,ie]
                    checked+=1
    return dict(status='passed',prespecified_endpoint_repetitions=sorted({0,cfg['K']-1}),independently_audited_private_assignments=checked,
                literal_unsquared_energy_formula_verified=True,ordinary_same_reports_and_report_specific_null_verified=True,
                bank_group_sizes_and_ge_plus_one_rule_verified=True,cohort_W1_and_swaps_verified=True,
                existing_calibration_and_row_equivariance_tests='see regression evidence')


def write_report(directory,output,manifest,df,summary,checks,figures):
    cfg=manifest['design'];g=json.loads((directory/'geometry.json').read_text());tables=output/'tables';tables.mkdir(exist_ok=True)
    rank=summary[(summary.procedure=='rank_energy')&summary.mechanism.isin(sr.ARMS)]
    refs=summary[(summary.procedure=='rank_energy')&~summary.mechanism.isin(sr.ARMS)].set_index('mechanism')
    text=['# Sonar private rank-energy experiment results','',
          f'Completed **{cfg["phase"]}** study `{manifest["fingerprint"]}`. The exact nonprivate squared-cost ranks have T={checks["nonprivate_T"]:.6f}, primary p={checks["nonprivate_primary_p"]:.4f}, and **{"reject" if checks["nonprivate_reject"] else "do not reject"}** at 0.05. Their p-values across the {cfg["B"]} public banks range from {checks["nonprivate_p_min"]:.4f} to {checks["nonprivate_p_max"]:.4f}. This is the actual comparator used below; no published outcome was forced.','',
          '## Setup and privacy boundary','',
          'All 208 UCI Sonar vectors retain their original 60 signal-energy coordinates in [0,1]. Every archived row was checked against UCI sonar.all-data; stored label 1 is rock and 2 is metal. There are 111 metal and 97 rock observations, with public labels and a fixed roster. Both cohorts’ measurements are confidential and are pooled on the private matching side. No cohort acts as the public OT sample. No RMS normalization, estimated standardization, feature selection, training split or classifier was used.','',
          f'The fixed public reference is the first 208 unscrambled Halton points indexed 1 through 208 in the first 60 prime bases. Its measured affine rank is **{g["public_rank"]}**, diameter {g["public_diameter"]:.6f}, radius {g["public_radius"]:.6f}, and directional-width certificate {g["coordinate_width_certificate"]:.6f}. Public PCA basis, widths, singular values and certificates are saved. Every output is a complete bijection to all 208 slots, optimized on squared Euclidean scores. Labels never enter scores, calibration or constraints. Independent row/column priorities resolve ties equivariantly in law.','',
          f'The nonprivate optimum total squared cost is {g["optimum_total_cost"]:.6f}; the exact uniform-permutation expectation is {g["random_expected_total_cost"]:.6f}. The smallest alternative assignment cost gap is {g["minimum_alternative_optimum_gap"]:.6g}, against numerical tie tolerance {g["optimum_tie_tolerance"]:.3g}; numerical uniqueness is {g["numerically_unique_optimum"]}. Agreement diagnostics use this one saved comparator throughout.','',
          'Each execution satisfies pure metric privacy with likelihood ratio at most exp(epsilon * sum_i ||r_i-r_i\u2032||_2). For one vector displaced by rho in original signal units, the bound is exp(epsilon*rho); there is no extra division of epsilon by 208. Gumbel protects only the final curator-released assignment; the other six arms protect reusable local reports. Energy tests on ranks, or on the same ambient reports, are postprocessing. Releasing multiple benchmark executions on confidential data would require composition. Raw data, unit noise and experimenter diagnostics in this research archive are not a private release.','',
          'The seven arms are ambient radial Laplace, ambient iid Laplace, public-span radial Laplace, coordinate allocation, centered squared-cost row infinity-norm noise, centered squared-cost row iid Laplace noise, and direct squared-cost Gumbel. Existing full-budget calibrations are retained. No blockwise, folded, anchor, selector or additional private arm is present.','',
          '| epsilon | exp(epsilon × 0.01) | exp(epsilon × 0.05) | exp(epsilon × 0.1) |','|---:|---:|---:|---:|']
    for epsilon in cfg['epsilon']:text.append('| '+f'{epsilon:g}'+' | '+' | '.join(f'{np.exp(epsilon*r):.3f}' for r in (.01,.05,.1))+' |')
    text+=['','## Conditional rejection frequencies','',
           f'For each arm and budget there are **K={cfg["K"]} independent privacy executions** on the same fixed cohort. The {cfg["B"]} independent public banks each contain J={cfg["J"]} uniform allocations with exactly 111 metal and 97 rock labels. Bank 1 was primary in advance. The unsquared-distance biased energy statistic includes zero diagonals and uses T=ab/N times energy. P-values use (1 + count(null T ≥ observed T))/(J+1); reject when p≤0.05. Assignments remain fixed during testing.','',
           'The table gives primary-bank rejection percentages. Figure intervals are pointwise 95% Wilson intervals over privacy repetitions conditional on bank 1. These are **conditional rejection frequencies over privacy noise**, not population statistical power or uncertainty over new cohorts.','',
           '| Mechanism | '+' | '.join(f'ε={e:g}' for e in cfg['epsilon'])+' |','|---|'+'---:|'*len(cfg['epsilon'])]
    for arm in sr.ARMS:
        part=rank[rank.mechanism==arm].set_index('epsilon')
        text.append('| '+LABELS[arm]+' | '+' | '.join(f'{100*part.loc[e,"rejection_frequency"]:.1f}%' for e in cfg['epsilon'])+' |')
        columns=['epsilon','rejection_frequency','rejection_low','rejection_high','rank_T_mean','primary_p_median','normalized_regret_mean','swaps_mean','Delta_mean']
        lines=['# '+LABELS[arm],'','Frequency intervals condition on the primary bank and the fixed Sonar cohort. Other columns are privacy-noise means, except the explicitly labeled p-value median.','',
               '| epsilon | Reject % [95% Wilson] | Mean T | Median p | Mean normalized regret | Mean swaps | Mean Delta |','|---:|---:|---:|---:|---:|---:|---:|']
        for row in part.reset_index().sort_values('epsilon').itertuples():
            lines.append(f'| {row.epsilon:g} | {100*row.rejection_frequency:.1f} [{100*row.rejection_low:.1f}, {100*row.rejection_high:.1f}] | {row.rank_T_mean:.3f} | {row.primary_p_median:.3f} | {row.normalized_regret_mean:.3f} | {row.swaps_mean:.2f} | {row.Delta_mean:.3f} |')
        (tables/(arm+'.md')).write_text('\n'.join(lines)+'\n')
    random=refs.loc['uniform_random']
    text+=['',f'The shared input-independent random-rank reference rejects in {int(random.rejection_count)}/{cfg["K"]} draws ({100*random.rejection_frequency:.1f}%; Wilson interval {100*random.rejection_low:.1f}–{100*random.rejection_high:.1f}%). It is one reference reused across epsilon. Nonprivate ranks are a fixed reference rather than K artificial repetitions. Unit private noise is deliberately shared across epsilon; channels and repetitions are independent, apart from any explicitly recorded equivalent endpoint. No epsilon values are pooled as extra trials.','',
           '![Primary rejection frequencies](figures/rank_rejection.png)','',
           '## Same-report ordinary energy comparison','',
           'For both ambient mechanisms, the ordinary energy test uses exactly the same released coordinates that supply squared rank matching. Its null distances and statistics are recomputed for each noisy dataset using J primary label allocations; it cannot use the universal Q-grid null statistics. The allocations themselves can be shared, as they are public and independent of measurements. The comparison adds no new privacy noise or private release. Paired decision differences and discordant counts are saved.','',
           '![Ordinary versus rank energy](figures/ordinary_report_comparison.png)','',
           '## Distributions and calibration-bank variation','',
           'Statistic and p-value bands describe distributions of individual private outputs, not uncertainty of their estimated means. Matching uses squared distance while energy uses unsquared distance.','',
           '![Statistic distributions](figures/rank_energy_distributions.png)','',
           '![P-value distributions](figures/rank_pvalue_distributions.png)','',
           'The next table fixes epsilon=16 and holds each private output fixed across all banks. Bank frequency ranges and standard deviations measure permutation Monte Carlo sensitivity; they do not create additional private repetitions or Wilson sample sizes. Near-threshold examples are selected for illustration only, one per arm at epsilon16, by closest primary p to 0.05 and then lowest draw index. Neither bank choice nor the primary decision changes.','',
           '| Mechanism | Primary reject % | Across-bank range % | Bank SD, percentage points | Fixed-output decision disagreement % |','|---|---:|---:|---:|---:|']
    for row in rank[rank.epsilon==16.].itertuples():
        text.append(f'| {LABELS[row.mechanism]} | {100*row.rejection_frequency:.1f} | {100*row.bank_frequency_min:.1f}–{100*row.bank_frequency_max:.1f} | {100*row.bank_frequency_sd:.2f} | {100*row.fixed_output_decision_disagreement_fraction:.1f} |')
    null=pd.read_csv(output/'null_calibration.csv');primary=null[null.bank==1].iloc[0]
    text+=['','![Bank robustness](figures/bank_robustness.png)','',
           '![Fixed-output p-values](figures/fixed_output_bank_pvalues.png)','',
           f'The separate conditional null check uses H={cfg["H"]} fresh public label allocations independent of all banks. Primary-bank null rejection is {int(primary.rejections)}/{cfg["H"]} ({100*primary.rejection_frequency:.1f}%; Wilson interval {100*primary.low:.1f}–{100*primary.high:.1f}%). Across banks, null rejection frequencies range from {100*null.rejection_frequency.min():.1f}% to {100*null.rejection_frequency.max():.1f}%. These same null cases are assessed across banks and are common to permutation-valued rank outputs, not method-specific evidence.','',
           '![Null calibration](figures/null_calibration.png)','',
           '## Matching and cohort-allocation diagnostics','',
           'True total squared regret, normalized regret, cohort allocation and their association with testing are NONPRIVATE experimenter diagnostics. They never tune an analyst mechanism or decision. Regret divides by exact RAND minus OPT only within this fixed matching problem; absolute totals remain available and values above the random level are retained.','',
           '![Squared regret](figures/squared_regret.png)','',
           'The minimum cross-cohort swap count is the number of excess metal slots relative to the nonprivate allocation, ignoring free within-cohort rearrangements. Normalized swaps are s/97. Geometric discrepancy Delta is W1(metal private ranks, metal nonprivate ranks) plus the analogous rock W1 on unsquared grid distances. Common rank atoms are canceled and the remaining exact metric assignment is solved; this equals the full uniform-cohort W1 problem.','',
           'For every saved output, the code checks |T−T(nonprivate)| ≤ (4ab/N) Delta. This is a continuity bound, not a guarantee of preserving a threshold decision.','',
           '![Cohort discrepancies](figures/cohort_discrepancy.png)','',
           '![Diagnostic relationships](figures/diagnostic_relationships.png)','',
           '## Interpretation of this completed cohort','']
    final=rank[rank.epsilon==16.].sort_values('rejection_frequency',ascending=False)
    best=final.iloc[0];worst=final.iloc[-1]
    text+=[f'At epsilon=16, the largest observed conditional rejection frequency is {100*best.rejection_frequency:.1f}% for {LABELS[best.mechanism]}, compared with {100*worst.rejection_frequency:.1f}% for {LABELS[worst.mechanism]} and {100*random.rejection_frequency:.1f}% for shared uniform ranks. These rankings describe this fixed cohort and bank. All seven budgets, unfavorable regimes and paired same-report comparisons remain visible.','',
           f'The observed nonprivate rank evidence {"rejects" if checks["nonprivate_reject"] else "does not reject"}; private preservation must be interpreted against that actual comparator. Mean normalized squared regret at epsilon16 ranges from {final.normalized_regret_mean.min():.3f} to {final.normalized_regret_mean.max():.3f}. A lower matching cost need not allocate rank regions to the two public cohorts in the same way, and neither swap counts nor Delta alone guarantees a rejection. The saved per-setting associations and joint diagnostic figure expose those distinctions without treating correlated epsilon outputs as independent cohorts.','',
           f'The largest fixed-output disagreement fraction across all arm/budget settings is {100*rank.fixed_output_decision_disagreement_fraction.max():.1f}%. The largest across-bank rejection-frequency range is {100*(rank.bank_frequency_max-rank.bank_frequency_min).max():.1f} percentage points. This documents sensitivity to permutation Monte Carlo randomness separately from the K privacy repetitions. The design, J, bank count and epsilon grid were not changed in response. Near-threshold p-values should be read with that recorded sensitivity.','',
           '## Validation, record and reproduction','',
           f'The canonical record contains {checks["private_draw_rows"]:,} private rank outputs, {checks["ambient_ordinary_rows"]:,} same-report ordinary-energy results, {cfg["K"]:,} shared random-rank references and one nonprivate reference. Source semantics, all source/preparation/unit SHA256 checksums, complete bijections, squared-cost stability and the W1 statistic bound are validated. An independent endpoint audit reconstructs true costs, literal energy sums and p-values, including the report-specific ordinary null. Pilot draws have separate seeds and directories.','',
           'Frozen numerical libraries and source are recorded. The explicit common profile fixes OpenBLAS to NEHALEM, disables optional NumPy dispatch groups and uses one numerical thread, permitting all compatible SMP CPU families. Runtime and host measurements are separate from deterministic scientific tables and figures.','',
           '- [Complete canonical numerical table](analysis_table.csv.gz)',
           '- [All conditional summaries and individual-output quantiles](summary.csv)',
           '- [Bank-specific frequencies](bank_rejection_frequencies.csv)',
           '- [Fixed-output near-threshold p-values](near_threshold_examples.csv)',
           '- [Same ambient report comparisons](ambient_report_comparison.csv)',
           '- [Null calibration](null_calibration.csv)',
           '- [Within-setting diagnostic associations](diagnostic_associations.csv)',
           '- [Public geometry and certificates](geometry.json)',
           '- [Manifest](manifest.json)',
           '- [Canonical verification](verification.json)',
           '- [Independent audit](audit_verification.json)',
           '- [Dictionary](data_dictionary.json)',
           '- [Exact reproduction entry point](reproduce.md)','',
           'Readable method tables:','']
    for arm in sr.ARMS:text.append(f'- [{LABELS[arm]}](tables/{arm}.md)')
    text+=['','Figures and PDF exports:','']
    for fig in figures:text.append(f'- [{fig["name"]}](figures/{fig["name"]}.png) · [PDF](figures/{fig["name"]}.pdf): {fig["caption"]}')
    text+=['','Sources: [UCI Sonar](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks), [Deb and Sen](https://arxiv.org/abs/1909.08733), and [authors’ reference implementation](https://github.com/NabarunD/MultiDistFree). The study follows the specified Halton grid rather than the reference implementation’s default grid.','']
    document='\n'.join(text)
    (output/'report.md').write_text(document);(output/'report.html').write_text(basic_html(document))
    for link in re.findall(r'\]\(([^)]+)\)',document):
        if not link.startswith(('http:','https:','#')) and not (output/link).exists():raise RuntimeError('STOP: missing report asset '+link)


def reproduction(directory,output,manifest):
    (output/'reproduce.md').write_text(f'''# Exact Sonar rank-study reproduction

Scientific fingerprint: `{manifest['fingerprint']}`. The source snapshot, input
files, all unit noise channels, 49,000 permutations, same ambient reports and
ordinary-null statistics are archived under the completed production run.
The public calibration allocations, their T values and independent null cases
are saved separately. Pilot and production use different seed namespaces.

Use the archived CPython runtime and numerical environment on a compatible
Linux/x86_64 CRC compute node. Numerical profile (set before importing NumPy):

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PRIVATE_MATCHING_NUM_THREADS=1 OPENBLAS_CORETYPE=NEHALEM
export NPY_DISABLE_CPU_FEATURES=X86_V3,X86_V4,AVX512_ICL,AVX512_SPR
export PYTHONPATH=RUN/source_snapshot/src
```

The self-contained entry point is `source_snapshot/slurm/sonar_rank_submit.sh`;
it discovers live Slurm running/submission limits and active allocations. Set
SONAR_RESULTS_ROOT to a fresh results root for an independent full rerun and
SONAR_SOURCE_DATA to the archived source directory. It submits preparation,
disjoint worker shards and dependent validation/reporting. All scientific seeds
come from explicit phase/repetition/channel keys, never scheduling order.

Direct compute-node entry points:

```bash
python -m private_matching study --study sonar_rank --results-root NEW_RESULTS \\
  --source-data ARCHIVED_SOURCES --stage all
python -m private_matching analysis-table --input RUN --output NEW_EXPORT
```

The first command performs a serial complete rerun on compute; use the archived
launcher for maximum permitted CRC parallelism. `--pilot` changes only the phase,
K to 3 and H to 100, retaining all seven budgets and all 20 banks of 9999.
The canonical gzip uses a zero timestamp and fixed row order; figures omit PDF
creation dates. Runtime and placement are separate nondeterministic records.
Reproducing public banks or p-values never redraws private reports. Replaying
ordinary energy recomputes its report-dependent null on saved reports.
''')


def finalize(directory):
    directory=Path(directory);output=directory/'analysis_handoff'
    manifest,df,summary,checks=export(directory,output)
    evidence=audit(directory,manifest);atomic_json(output/'audit_verification.json',evidence)
    figures=plot_all(output,manifest,df,summary,checks)
    reproduction(directory,output,manifest);write_report(directory,output,manifest,df,summary,checks,figures)
    atomic_json(directory/'COMPLETE.json',checks)
    print('SONAR HANDOFF:',output,flush=True)
