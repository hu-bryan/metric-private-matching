"""Fixed-cohort Sonar squared rank matching and public-bank rank-energy tests.

Only the seven established squared-cost laws are adapted here. Labels enter
evaluation and tests, never noise calibration, scores, or assignment constraints.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import resource
import time
import zipfile

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from . import clustering_benchmark as cb
from .integral_mechanisms import Geometry,bound,solve,supnorm_unit
from .integral_study import atomic_json,save_npz,seed
from .login_guard import refuse_login_compute
from .mechanisms import sample_euclidean_laplace

NAME='sonar_rank_energy_v1'
NAMESPACE='sonar-rank-energy-20261112-v1'
ARMS=('ambient_radial','ambient_iid','span_radial','coordinate_allocated','cost_infinity','cost_iid','gumbel')
EPSILON=(.25,.5,1.,2.,4.,8.,16.)
PROFILE={'OPENBLAS_CORETYPE':'NEHALEM','NPY_DISABLE_CPU_FEATURES':'X86_V3,X86_V4,AVX512_ICL,AVX512_SPR',
         'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
SOURCE_FILES=('sonar.data.gz','sonar.labels0.gz','sonar.txt','uci_original.zip')
DEFAULT_SOURCES='/vast/arahimian/brh314/metric_private_covariate_matching/results/sonar_rank_sources_v1'


def design(pilot=False):
    return dict(phase='pilot' if pilot else 'production',K=3 if pilot else 1000,
                N=208,d=60,a=111,b=97,epsilon=list(EPSILON),arms=list(ARMS),
                J=9999,B=20,H=100 if pilot else 1000,alpha=.05,primary_bank=1,
                objective_p=2,energy_distance_p=1,normalization='none; original signal units',
                public_reference='unscrambled Halton; first 60 primes; indices 1 through 208',
                tie_rule='independent uniform row and column priorities before exact assignment; label blind; equivariant in law',
                noise_protocol='independent channels and repetition IDs; shared unit noise across epsilon; explicit equivalent-endpoint reuse only',
                ordinary_energy='same ambient reports; primary label allocations shared, null statistics recomputed on each noisy dataset',
                null_bank_protocol='20 independent public allocation banks; bank 1 primary; no B*K pooling',
                random_reference='one input-independent uniform permutation per repetition, reused across epsilon',
                numerical_profile=PROFILE,energy_batch_rows=512)


def rng(*keys):return np.random.default_rng(seed(NAMESPACE,*keys))


def hardware():
    models=[line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')]
    return dict(host=platform.node(),cpu_model=models[0] if models else None,
                job_id=os.environ.get('SLURM_JOB_ID'),array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),profile={k:os.environ.get(k) for k in PROFILE})


def package_hashes():
    package=Path(__file__).parent
    return {p.name:cb.sha256(p) for p in sorted(package.glob('*.py'))}


def validate_sources(root):
    root=Path(root);R=np.loadtxt(root/'sonar.data.gz');stored=np.loadtxt(root/'sonar.labels0.gz',dtype=int)
    with zipfile.ZipFile(root/'uci_original.zip') as source:
        rows=np.genfromtxt(io.StringIO(source.read('sonar.all-data').decode()),delimiter=',',dtype=str)
        names=source.read('sonar.names').decode()
    if not np.array_equal(R,rows[:,:60].astype(float)):
        raise RuntimeError('STOP: archived Sonar rows do not exactly match UCI originals')
    semantic=rows[:,60]
    mapping={str(v):np.unique(semantic[stored==v]).tolist() for v in np.unique(stored)}
    if mapping!={'1':['R'],'2':['M']} or R.shape!=(208,60) or not np.isfinite(R).all() or not ((R>=0)&(R<=1)).all():
        raise RuntimeError('STOP: Sonar source shape, units, or semantic label verification failed')
    metal=semantic=='M'
    if int(metal.sum())!=111 or int((~metal).sum())!=97:raise RuntimeError('STOP: wrong public roster')
    files={name:dict(sha256=cb.sha256(root/name),bytes=(root/name).stat().st_size) for name in SOURCE_FILES}
    return R,metal,stored,dict(files=files,semantic_mapping=mapping,verification='every numeric row and semantic label checked against UCI sonar.all-data; mapping not inferred from counts',
                              official_url='https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks',
                              original_zip_url='https://archive.ics.uci.edu/static/public/151/connectionist+bench+sonar+mines+vs+rocks.zip',
                              names_sha256=hashlib.sha256(names.encode()).hexdigest())


def public_grid():
    # SciPy index zero is the all-zero point and is deliberately excluded.
    return qmc.Halton(d=60,scramble=False).random(209)[1:]


def prioritized_solve(scores,generator):
    n=len(scores);rows=generator.permutation(n);cols=generator.permutation(n)
    selected=solve(scores[rows][:,cols]);pi=np.empty(n,dtype=np.uint16);pi[rows]=cols[selected]
    return pi


def label_allocations(count,n,a,generator):
    labels=np.zeros((count,n),dtype=bool)
    for row in labels:row[generator.choice(n,a,replace=False)]=True
    return labels


def centered_distances(D):
    means=D.mean(axis=1)
    return D-means[:,None]-means[None,:]+means.mean()


def energy_statistics(D,allocations,a,b,batch=512):
    """Exactly the biased energy statistic T, using sum-zero label contrasts.

    Double centering changes no mathematical statistic and avoids unnecessary
    cancellation for large ambient-noise distances. Diagonals remain included.
    """
    allocations=np.asarray(allocations)
    if allocations.ndim==1:allocations=allocations[None,:]
    n=a+b;Dc=centered_distances(np.asarray(D,float));result=np.empty(len(allocations))
    for start in range(0,len(allocations),batch):
        H=allocations[start:start+batch].astype(float)-a/n
        result[start:start+len(H)]=-n/(a*b)*np.sum((H@Dc)*H,axis=1)
    return result


def bank_pvalues(statistics,sorted_banks):
    observed=np.asarray(statistics)
    J=sorted_banks.shape[1]
    return np.stack([(1+J-np.searchsorted(bank,observed,side='left'))/(J+1) for bank in sorted_banks],axis=-1)


def slot_membership(pi,metal):
    mask=np.zeros(len(pi),dtype=bool);mask[pi[metal]]=True
    return mask


def cohort_discrepancy(pi,nonprivate_pi,metal,D):
    """Exact W1: cancel common ranks, then transport excess to missing ranks.

    Triangle inequality permits cancellation of shared atoms in a metric W1
    problem. The rock problem is the reversed same excess/missing transport,
    with weight 1/b instead of 1/a. Thus one small exact solve suffices.
    """
    actual=slot_membership(pi,metal);target=slot_membership(nonprivate_pi,metal)
    excess=np.flatnonzero(actual&~target);missing=np.flatnonzero(target&~actual)
    s=len(excess)
    if len(missing)!=s:raise RuntimeError('cohort mass mismatch')
    distance=0.
    if s:
        costs=D[excess][:,missing];permutation=solve(costs)
        distance=float(costs[np.arange(s),permutation].sum())
    a=int(metal.sum());b=len(metal)-a
    return s,distance/a,distance/b


def channels(geo,phase,draw):
    n,d=geo.Y.shape;r=geo.rank
    generator=lambda arm:rng(phase,'private',draw,arm,'unit-noise')
    values={
        'ambient_radial':sample_euclidean_laplace(n,d,1.,generator('ambient_radial'))[0],
        'ambient_iid':np.sqrt(d)*generator('ambient_iid').laplace(size=(n,d)),
        'cost_infinity':supnorm_unit(n,generator('cost_infinity')),
        'cost_iid':generator('cost_iid').laplace(size=(n,n)),
        'gumbel':generator('gumbel').gumbel(size=(n,n))}
    values['span_radial']=values['ambient_radial'] if r==d else sample_euclidean_laplace(n,r,1.,generator('span_radial'))[0]
    rates=geo.widths**(1/3)/np.sqrt(np.sum(geo.widths**(2/3)))
    values['coordinate_allocated']=generator('coordinate_allocated').laplace(size=(n,r))/rates
    return values


def squared_scores(geo,noise,arm,epsilon):
    """Adapters of integral_mechanisms' established squared-cost laws."""
    public=geo.Y-geo.center
    if arm in ('ambient_radial','ambient_iid'):perturbation=-2*noise[arm]@public.T
    elif arm=='span_radial':
        perturbation=-2*noise[arm]@(public.T if geo.rank==geo.Y.shape[1] else geo.V.T)
    elif arm=='coordinate_allocated':perturbation=-2*noise[arm]@geo.V.T
    elif arm=='gumbel':perturbation=-2*geo.diameter*noise[arm]
    else:
        cert=bound(geo,arm,2,epsilon,'benchmark')
        perturbation=noise[arm]*cert['L_infinity' if arm=='cost_infinity' else 'L_one']
    return geo.base2+perturbation/epsilon


def relocation(manifest):
    if not Path(manifest['source_data']).exists():manifest['source_data']=manifest['source_data'].replace('/vast/','/ix1/',1)
    return manifest


def verify_runtime(manifest):
    if any(os.environ.get(k)!=v for k,v in PROFILE.items()):raise RuntimeError('STOP: Sonar numerical profile differs')
    if cb.environment()!=manifest['versions'] or package_hashes()!=manifest['package_sha256']:
        raise RuntimeError('STOP: frozen Sonar source/environment drift')
    for name,record in manifest['sources']['files'].items():
        if cb.sha256(Path(manifest['source_data'])/name)!=record['sha256']:raise RuntimeError('STOP: Sonar source checksum drift')


def prepare(results_root,source_data,pilot=False):
    refuse_login_compute('prepare the Sonar rank study')
    R,metal,stored,sources=validate_sources(source_data)
    payload=dict(kind=NAME,design=design(pilot),sources=sources,package_sha256=package_hashes(),versions=cb.environment())
    fp=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:24]
    manifest=dict(payload,fingerprint=fp,source_data=str(Path(source_data).resolve()))
    verify_runtime(manifest)
    directory=Path(results_root)/NAME/manifest['design']['phase']/fp
    directory.mkdir(parents=True,exist_ok=True);(directory/'units').mkdir(exist_ok=True)
    if (directory/'manifest.json').exists():
        old=relocation(json.loads((directory/'manifest.json').read_text()))
        if old!=manifest:raise RuntimeError('STOP: Sonar manifest conflict')
        if (directory/'PREPARED.json').exists():
            check_prepared(directory);return directory,manifest
    atomic_json(directory/'manifest.json',manifest)
    cfg=manifest['design'];Q=public_grid();geo=Geometry.build(R,Q)
    pi=prioritized_solve(geo.C2,rng('nonprivate','assignment-priorities'))
    optimum=float(geo.C2[np.arange(cfg['N']),pi].sum());random=float(geo.C2.sum()/cfg['N'])
    gap=cb.gap_info(optimum,random)
    if gap['status']=='zero_gap':raise RuntimeError('STOP: zero Sonar assignment gap needs explicit review')
    D=cdist(Q,Q);T=float(energy_statistics(D,slot_membership(pi,metal),cfg['a'],cfg['b'])[0])
    # Determine whether agreement diagnostics depend on a numerically tied optimum.
    alternatives=[]
    for i in range(cfg['N']):
        changed=geo.C2.copy();changed[i,pi[i]]=np.inf;other=solve(changed)
        alternatives.append(float(geo.C2[np.arange(cfg['N']),other].sum()-optimum))
    tie_tolerance=64*np.finfo(float).eps*max(1.,abs(optimum))
    geometry=dict(geo.public_metadata(),coordinate_widths=geo.widths.tolist(),singular_values=geo.singular.tolist(),
                  optimum_total_cost=optimum,random_expected_total_cost=random,gap=gap,
                  nonprivate_T=T,nonprivate_E=T*cfg['N']/(cfg['a']*cfg['b']),
                  minimum_alternative_optimum_gap=min(alternatives),optimum_tie_tolerance=tie_tolerance,
                  numerically_unique_optimum=bool(min(alternatives)>tie_tolerance),
                  bounds={arm:[bound(geo,arm,2,e,'benchmark') for e in cfg['epsilon']] for arm in ARMS})
    # JSON encodes the uncapped squared certificate as null, never Infinity.
    for records in geometry['bounds'].values():
        for cert in records:
            for key,value in cert.items():
                if isinstance(value,float) and not np.isfinite(value):cert[key]=None
    atomic_json(directory/'geometry.json',geometry)
    save_npz(directory/'inputs.npz',R=R,Q=Q,metal=metal,stored_labels=stored,nonprivate_permutation=pi,
             public_B=geo.B,public_center=geo.center,public_widths=geo.widths,public_singular=geo.singular,
             true_squared_cost=geo.C2,grid_distances=D)
    allocations=[];statistics=[]
    for bank in range(cfg['B']):
        masks=label_allocations(cfg['J'],cfg['N'],cfg['a'],rng(cfg['phase'],'public-bank',bank))
        allocations.append(masks);statistics.append(energy_statistics(D,masks,cfg['a'],cfg['b']))
    banks=np.asarray(statistics);ordered=np.sort(banks,axis=1)
    save_npz(directory/'banks.npz',allocations=np.asarray(allocations),statistics=banks,sorted_statistics=ordered,
             nonprivate_pvalues=bank_pvalues(T,ordered))
    null_cases=label_allocations(cfg['H'],cfg['N'],cfg['a'],rng(cfg['phase'],'independent-null-cases'))
    null_T=energy_statistics(D,null_cases,cfg['a'],cfg['b'])
    save_npz(directory/'null_check.npz',allocations=null_cases,statistics=null_T,pvalues=bank_pvalues(null_T,ordered))
    files={name:dict(sha256=cb.sha256(directory/name),bytes=(directory/name).stat().st_size)
           for name in ('inputs.npz','banks.npz','null_check.npz','geometry.json')}
    atomic_json(directory/'PREPARED.json',dict(files=files,hardware=hardware()))
    distributions=sorted(importlib.metadata.distributions(),key=lambda d:d.metadata['Name'].lower())
    (directory/'requirements.freeze.txt').write_text(''.join(f'{d.metadata["Name"]}=={d.version}\n' for d in distributions))
    print('PUBLIC GRID RANK:',geo.rank,'NONPRIVATE T:',T,'PRIMARY P:',float(bank_pvalues(T,ordered)[0]),flush=True)
    return directory,manifest


def check_prepared(directory):
    marker=json.loads((directory/'PREPARED.json').read_text())
    for name,record in marker['files'].items():
        if cb.sha256(directory/name)!=record['sha256']:raise RuntimeError('STOP: prepared Sonar artifact checksum mismatch')


def load_context(directory,manifest):
    with np.load(directory/'inputs.npz') as saved:inputs={name:saved[name] for name in saved.files}
    with np.load(directory/'banks.npz') as saved:
        banks=dict(sorted=saved['sorted_statistics'],primary_allocations=saved['allocations'][0])
    geometry=json.loads((directory/'geometry.json').read_text())
    return dict(inputs=inputs,banks=banks,geometry=geometry,geo=Geometry.build(inputs['R'],inputs['Q']))


def checkpoint(directory,draw,fp,verify=True):
    marker=directory/'units'/f'draw{draw:04d}.json'
    if not marker.exists():return None
    meta=json.loads(marker.read_text())
    if meta['fingerprint']!=fp or meta['draw']!=draw:raise RuntimeError('STOP: Sonar checkpoint identity mismatch')
    if verify:
        for name,record in meta['files'].items():
            if cb.sha256(directory/'units'/name)!=record['sha256']:raise RuntimeError('STOP: Sonar checkpoint checksum mismatch')
    return meta


def run_draw(directory,manifest,draw,context=None):
    if checkpoint(directory,draw,manifest['fingerprint']) is not None:return
    started=time.perf_counter();cfg=manifest['design'];context=context or load_context(directory,manifest)
    data=context['inputs'];geo=context['geo'];banks=context['banks'];g=context['geometry']
    metal=data['metal'];D=data['grid_distances'];reference=data['nonprivate_permutation'];n=cfg['N']
    noise=channels(geo,cfg['phase'],draw);shape=(len(ARMS),len(cfg['epsilon']))
    result={name:np.empty(shape) for name in ('total_cost','total_regret','normalized_regret','rank_T','swaps','W1_M','W1_R','Delta','statistic_change','realization_bound')}
    result['permutations']=np.empty((*shape,n),dtype=np.uint16)
    result['rank_pvalues']=np.empty((*shape,cfg['B']))
    result['ambient_reports']=np.empty((2,len(cfg['epsilon']),n,cfg['d']))
    result['ordinary_T']=np.empty((2,len(cfg['epsilon'])))
    result['ordinary_pvalues']=np.empty((2,len(cfg['epsilon'])))
    result['ordinary_null_statistics']=np.empty((2,len(cfg['epsilon']),cfg['J']))
    aliases={}
    for ia,arm in enumerate(ARMS):
        for ie,epsilon in enumerate(cfg['epsilon']):
            if arm=='span_radial' and geo.rank==cfg['d']:
                pi=result['permutations'][0,ie].copy();aliases[arm]='ambient_radial'
            else:
                scores=squared_scores(geo,noise,arm,epsilon)
                pi=prioritized_solve(scores,rng(cfg['phase'],'private',draw,arm,epsilon,'assignment-priorities'))
            if not np.array_equal(np.sort(pi),np.arange(n)):raise RuntimeError('STOP: invalid Sonar permutation')
            result['permutations'][ia,ie]=pi
            cost=float(geo.C2[np.arange(n),pi].sum());regret=cost-g['optimum_total_cost']
            scores=squared_scores(geo,noise,arm,epsilon)
            realization=float(np.ptp(scores-geo.base2,axis=1).sum())
            if regret < -1e-9 or regret>realization+1e-8*max(1.,realization):raise RuntimeError('STOP: squared stability certificate failed')
            T=float(energy_statistics(D,slot_membership(pi,metal),cfg['a'],cfg['b'])[0])
            swaps,wM,wR=cohort_discrepancy(pi,reference,metal,D);delta=wM+wR
            change=abs(T-g['nonprivate_T'])
            if change>4*cfg['a']*cfg['b']/n*delta+1e-10:raise RuntimeError('STOP: cohort W1 energy bound failed')
            for name,value in dict(total_cost=cost,total_regret=regret,normalized_regret=regret/g['gap']['value'],rank_T=T,
                                   swaps=swaps,W1_M=wM,W1_R=wR,Delta=delta,statistic_change=change,realization_bound=realization).items():
                result[name][ia,ie]=value
            result['rank_pvalues'][ia,ie]=bank_pvalues(T,banks['sorted'])
            if ia<2:
                report=data['R']+noise[arm]/epsilon
                result['ambient_reports'][ia,ie]=report
                distances=cdist(report,report)
                obs=float(energy_statistics(distances,metal,cfg['a'],cfg['b'])[0])
                null=energy_statistics(distances,banks['primary_allocations'],cfg['a'],cfg['b'])
                result['ordinary_T'][ia,ie]=obs;result['ordinary_null_statistics'][ia,ie]=null
                result['ordinary_pvalues'][ia,ie]=(1+np.count_nonzero(null>=obs))/(cfg['J']+1)
    random=rng(cfg['phase'],'uniform-rank-reference',draw).permutation(n).astype(np.uint16)
    T=float(energy_statistics(D,slot_membership(random,metal),cfg['a'],cfg['b'])[0])
    swaps,wM,wR=cohort_discrepancy(random,reference,metal,D)
    cost=float(geo.C2[np.arange(n),random].sum())
    result.update(random_permutation=random,random_T=np.array(T),random_pvalues=bank_pvalues(T,banks['sorted']),
                  random_cost=np.array(cost),random_regret=np.array(cost-g['optimum_total_cost']),
                  random_swaps=np.array(swaps),random_Delta=np.array(wM+wR))
    result.update({'unit_noise_'+arm:value for arm,value in noise.items()})
    path=directory/'units'/f'draw{draw:04d}.npz';save_npz(path,**result)
    atomic_json(path.with_suffix('.json'),dict(fingerprint=manifest['fingerprint'],draw=draw,
                files={path.name:dict(sha256=cb.sha256(path),bytes=path.stat().st_size)},
                hardware=hardware(),elapsed_seconds=time.perf_counter()-started,
                peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),aliases=aliases,
                private_executions=len(ARMS)*len(cfg['epsilon']),ordinary_reuses_same_reports=True))
    print('COMPLETE DRAW',draw,f'{time.perf_counter()-started:.3f}s',flush=True)


def run_study(root,sources=None,pilot=False,stage='all',shard_index=None,shard_count=None,directory=None):
    refuse_login_compute('run the Sonar rank experiment')
    if stage in ('prepare','all'):directory,manifest=prepare(root,sources or DEFAULT_SOURCES,pilot)
    else:
        directory=Path(directory);manifest=relocation(json.loads((directory/'manifest.json').read_text()))
    print('Resolved Sonar rank results:',directory,flush=True)
    verify_runtime(manifest);check_prepared(directory)
    records=directory/'stage_records';records.mkdir(exist_ok=True)
    atomic_json(records/f'{stage}_{os.environ.get("SLURM_JOB_ID")}_{shard_index}.json',dict(stage=stage,hardware=hardware(),frozen_environment_checked=True))
    if stage in ('shard','all'):
        index=0 if shard_index is None else shard_index;count=1 if shard_count is None else shard_count
        if not 0<=index<count:raise ValueError('invalid Sonar shard')
        context=load_context(directory,manifest)
        for draw in range(index,manifest['design']['K'],count):run_draw(directory,manifest,draw,context)
    if stage in ('finalize','all'):
        from .sonar_rank_report import finalize
        finalize(directory)
    return directory
