"""Separately versioned n=64 epsilon-only benchmark extension and isometries."""
from __future__ import annotations
from functools import lru_cache
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.distance import cdist

from . import clustering_benchmark as cb
from .integral_mechanisms import Geometry
from .integral_study import atomic_json, save_npz, seed
from .login_guard import refuse_login_compute

NAME='clustering_benchmark_extension_v1'
NAMESPACE='clustering-benchmark-extension-20261112-v1'
ORIGINAL_FP='975562db8834d0cbf8bb0683'
HARDWARE_POLICY=dict(slurm_constraint='turin',cpu_family='AMD Turin',cluster='smp',
                     stages=['prepare','shard','finalize','regression','archived_environment_replay'],
                     maximum_concurrent_workers=125,numerical_threads_per_worker=1,
                     frozen_environment_check='exact source, library, numerical binary and NumPy CPU dispatch identity')
ORIGINAL=Path('/ix1/arahimian/brh314/metric_private_covariate_matching/results/clustering_benchmark_v1/production')/ORIGINAL_FP
DATASETS={'uci/wine':(178,13),'uci/statlog':(2310,19),'uci/sonar':(208,60),
          'uci/ionosphere':(351,34),'uci/glass':(214,9),'uci/ecoli':(336,7),
          'uci/yeast':(1484,8),'other/iris':(150,4),'mnist/fashion10k':(10000,784),
          'wut/circles':(4000,2)}
for family in ('g2mg','h2mg'):
    for dimension in (32,128):
        for spread in (20,50,80):
            DATASETS[f'{family}/{family}_{dimension}_{spread}']=(2048,dimension)
INVARIANT={1:('gumbel','anchor','cost_infinity','cost_iid'),
           2:('gumbel','span_radial','coordinate_allocated','cost_infinity','cost_iid')}
FLOAT_FIELDS=cb.FLOAT_FIELDS


@lru_cache(maxsize=2)
def load_dataset(root,dataset):
    path=Path(root)
    x=np.loadtxt(path/(dataset+'.data.gz'))
    labels=np.loadtxt(path/(dataset+'.labels0.gz'),dtype=int)
    if x.shape!=DATASETS[dataset] or labels.shape!=(len(x),) or not np.isfinite(x).all():
        raise RuntimeError(f'STOP: invalid extension source {dataset}: {x.shape}')
    return x,labels


def validate_sources(root):
    root=Path(root)
    tree=json.loads((root/'pinned_git_tree.json').read_text())
    if tree['sha']!=cb.COMMIT or tree['truncated']:
        raise RuntimeError('STOP: wrong/incomplete pinned source inventory')
    blobs={entry['path']:entry['sha'] for entry in tree['tree'] if entry['type']=='blob'}
    names=sorted({f'{dataset}.{ext}' for dataset in DATASETS for ext in ('txt','data.gz','labels0.gz')} |
                 {dataset.split('/')[0]+'/README.txt' for dataset in DATASETS})
    files={}
    for name in names:
        blob=(root/name).read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
        if actual!=blobs[name]: raise RuntimeError('STOP: immutable Git blob mismatch: '+name)
        files[name]=dict(sha256=cb.sha256(root/name),git_blob_sha1=actual,bytes=len(blob),
                        url=f'https://raw.githubusercontent.com/gagolews/clustering-data-v1/{cb.COMMIT}/{name}')
    files['pinned_git_tree.json']=dict(sha256=cb.sha256(root/'pinned_git_tree.json'),bytes=(root/'pinned_git_tree.json').stat().st_size,
                                      url=f'https://api.github.com/repos/gagolews/clustering-data-v1/git/trees/{cb.COMMIT}?recursive=1')
    inventory={}
    for dataset in DATASETS:
        x,labels=load_dataset(str(root),dataset)
        ids,counts=np.unique(labels,return_counts=True)
        inventory[dataset]=dict(shape=list(x.shape),stored_label_counts=dict(zip(map(str,ids),map(int,counts))),
                                noise_count=int(np.sum(labels==0)),noise_proportion=float(np.mean(labels==0)))
    original=json.loads((ORIGINAL/'manifest.json').read_text())
    for ext in ('data.gz','labels0.gz','txt'):
        name='wut/circles.'+ext
        if files[name]['sha256']!=original['sources']['files'][name]['sha256']:
            raise RuntimeError('STOP: Circles source changed relative to completed benchmark')
    return dict(version='v1.2.0',commit=cb.COMMIT,files=files,datasets=inventory,
                labels='labels0 reference partition; no semantic label conditioning',source_row_index='zero-based original source rows')


def settings():
    result=[]
    for dataset,(_,d) in DATASETS.items():
        family=dataset.split('/')[0]
        group=family if family in ('g2mg','h2mg') else ('embedding' if dataset=='wut/circles' else 'real')
        sid='native_'+dataset.replace('/','_')
        memberships=[dict(panel=group,condition='mixed')]
        if dataset=='other/iris': memberships.append(dict(panel='embedding',condition='mixed'))
        result.append(dict(setting_id=sid,dataset=dataset,n=64,d=d,condition='mixed',feature_order='native',
                           group=group,embedded=False,memberships=memberships,
                           imported_original=dataset=='wut/circles',native_setting_id=sid))
    for dataset in ('wut/circles','other/iris'):
        result.append(dict(setting_id='embedded_'+dataset.replace('/','_')+'_d128',dataset=dataset,n=64,d=128,
                           condition='mixed',feature_order='isometric_zero_padding',group='embedding',embedded=True,
                           memberships=[dict(panel='embedding',condition='mixed')],imported_original=False,
                           native_setting_id='native_'+dataset.replace('/','_')))
    return result


def design(pilot=False):
    cfg=cb.design(pilot)
    cfg.update(schema=NAME,settings=settings(),seed_namespace=NAMESPACE,
               epsilon=[.25,4.,16.] if pilot else [.25,.5,1.,2.,4.,8.,16.],
               feature_seed=None,dimension_sweep=False,sample_size_sweep=False,label_condition_sweep=False,
               group_summary_rule='real, g2mg and h2mg separately; equal dataset weights within each; no overall mixture; controls have no pooled average',
               rho_definition='effective_rank / covariance_rank, alias of existing rank ratio',
               directional_width_definition='W_Q=(sum_k width_k**(2/3))**(3/2), public PCA coordinates, existing coordinate-allocation certificate',
               embedding=dict(map='x -> (x, 0,...,0) in R128',matrix='saved in geometry NPZ',
                              paired_rows='same source IDs, roles and order as native',
                              distance_tolerance='2e-12*max(1,max native pairwise distance)',
                              invariant_reused_arms={str(p):list(arms) for p,arms in INVARIANT.items()},
                              fold_rule='unchanged min(public rank+1,ambient d); no assumed fold invariance'),
               original_fingerprint=ORIGINAL_FP,original_policy='read-only; original Circles selections and production executions imported with provenance')
    return cfg


def source_hashes():
    package=Path(__file__).parent
    return {str(path.relative_to(package.parent)):cb.sha256(path) for path in sorted(package.glob('*.py'))}


def hardware_record():
    models=[line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
            if line.startswith('model name')]
    return dict(hostname=platform.node(),cpu_model=models[0] if models else None,
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                slurm_array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
                requested_constraint=os.environ.get('PM_EXT_CPU_CONSTRAINT','turin'))


def resolve(results_root,source_data,pilot=False):
    payload=dict(kind=NAME,design=design(pilot),hardware_policy=HARDWARE_POLICY,sources=validate_sources(source_data),
                 source_sha256=source_hashes(),versions=cb.environment(),
                 original=dict(fingerprint=ORIGINAL_FP,manifest_sha256=cb.sha256(ORIGINAL/'manifest.json'),
                               completion_sha256=cb.sha256(ORIGINAL/'COMPLETE.json')))
    fp=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:24]
    payload.update(fingerprint=fp,source_data=str(Path(source_data).resolve()),original_directory=str(ORIGINAL))
    return Path(results_root)/NAME/payload['design']['phase']/fp,payload


def verify_runtime(manifest):
    if source_hashes()!=manifest['source_sha256'] or cb.environment()!=manifest['versions']:
        raise RuntimeError('STOP: frozen source/environment drift')
    for name,record in manifest['sources']['files'].items():
        if cb.sha256(Path(manifest['source_data'])/name)!=record['sha256']:
            raise RuntimeError('STOP: extension source changed: '+name)
    if cb.sha256(ORIGINAL/'manifest.json')!=manifest['original']['manifest_sha256']:
        raise RuntimeError('STOP: original manifest changed')


def native_unit(cfg,unit):
    return next(u for u in cb.all_units(cfg) if u['setting_id']==unit['native_setting_id'] and
                (u['parent'],u['partition'])==(unit['parent'],unit['partition']))


def selected_source(manifest,unit):
    cfg=manifest['design']; dataset=unit['dataset']
    source,labels=load_dataset(manifest['source_data'],dataset)
    if dataset=='wut/circles' and cfg['phase']=='production':
        oldid=f'aggregate_wut_circles_parent{unit["parent"]:02d}_part{unit["partition"]}'
        oldmeta=json.loads((ORIGINAL/'units'/(oldid+'.json')).read_text())
        for name,record in oldmeta['files'].items():
            if cb.sha256(ORIGINAL/'units'/name)!=record['sha256']:
                raise RuntimeError('STOP: original Circles checkpoint corrupt')
        with np.load(ORIGINAL/'units'/(oldid+'.geometry.npz')) as saved:
            rec={key:saved[key] for key in saved.files if key.startswith(('q_','r_','maximum_','parent_'))}
        origin=dict(fingerprint=ORIGINAL_FP,instance_id=oldid,directory=str(ORIGINAL),
                    geometry_sha256=cb.sha256(ORIGINAL/'units'/(oldid+'.geometry.npz')))
    else:
        key=(NAMESPACE,cfg['phase'],dataset,unit['parent'])
        pool=np.random.default_rng(seed(*key,'aggregate-parent')).choice(len(source),128,replace=False)
        order=np.random.default_rng(seed(*key,'aggregate-partition',unit['partition'])).permutation(pool)
        qi,ri=order[:64],order[64:]
        rec=dict(q_source_ids=qi,r_source_ids=ri,q_stored_labels=labels[qi],r_stored_labels=labels[ri],
                 maximum_Q=qi,maximum_R=ri,parent_mixed=pool)
        origin=dict(fingerprint=manifest['fingerprint'],instance_id=unit['native_setting_id']+f'_parent{unit["parent"]:02d}_part{unit["partition"]}',
                    directory=None)
    qi,ri=rec['q_source_ids'],rec['r_source_ids']
    if len(np.unique(np.r_[qi,ri]))!=128: raise RuntimeError('STOP: overlapping matching source rows')
    # Match the original benchmark's explicit feature indexing, including array
    # layout, so normalization replays the original floating-point sum order.
    features=np.arange(source.shape[1])
    Q,R=source[qi][:,features],source[ri][:,features]
    rec['feature_ids']=np.arange(Q.shape[1])
    return Q,R,rec,origin


def prepared_instance(manifest,unit):
    Q,R,records,origin=selected_source(manifest,unit)
    extra=dict(execution_origin=origin,embedding=None,extension_execution_hardware=hardware_record())
    if unit['embedded']:
        E=np.pad(np.eye(Q.shape[1]),((0,0),(0,128-Q.shape[1])))
        qe,re=Q@E,R@E
        original=cdist(np.r_[Q,R],np.r_[Q,R])
        embedded=cdist(np.r_[qe,re],np.r_[qe,re])
        tol=2e-12*max(1.,float(original.max()))
        q,r,_,s=cb.standardize(Q,R); q2,r2,_,s2=cb.standardize(qe,re)
        normalized_error=float(np.max(np.abs(cdist(np.r_[q,r],np.r_[q,r])-cdist(np.r_[q2,r2],np.r_[q2,r2]))))
        if not np.array_equal(E@E.T,np.eye(Q.shape[1])) or np.max(np.abs(original-embedded))>tol or normalized_error>2e-12*max(1.,float(original.max()/s)):
            raise RuntimeError('STOP: isometric embedding distance preservation failed')
        g,h=Geometry.build(r,q),Geometry.build(r2,q2)
        if g.rank!=h.rank or not np.allclose(g.widths,h.widths,rtol=1e-10,atol=1e-12):
            raise RuntimeError('STOP: embedded public PCA widths/rank not equivalent')
        if not np.allclose(np.abs(g.B.T@E@h.B),np.eye(g.rank),rtol=1e-10,atol=1e-10):
            raise RuntimeError('STOP: PCA bases need review before coordinate-allocation reuse')
        extra['embedding']=dict(type='isometric_zero_padding',native_dimension=Q.shape[1],ambient_dimension=128,
                                native_instance_id=origin['instance_id'],native_fingerprint=origin['fingerprint'],
                                native_scale=s,embedded_scale=s2,raw_distance_max_error=float(np.max(np.abs(original-embedded))),
                                raw_distance_tolerance=tol,standardized_distance_max_error=normalized_error,
                                all_pairwise_QR_verified=True,public_widths_verified=True)
        records.update(embedding_matrix=E,native_feature_ids=records['feature_ids'],feature_ids=np.arange(128))
        Q,R=qe,re
    return (Q,R,records),extra


def import_circles(directory,manifest,unit):
    _,extra=prepared_instance(manifest,unit)
    origin=extra['execution_origin']; oldid=origin['instance_id']
    meta=json.loads((ORIGINAL/'units'/(oldid+'.json')).read_text())
    stem=directory/'units'/unit['instance_id']
    files={}
    for suffix in ('.npz','.geometry.npz'):
        dest=Path(str(stem)+suffix)
        shutil.copyfile(ORIGINAL/'units'/(oldid+suffix),dest)
        files[dest.name]=dict(bytes=dest.stat().st_size,sha256=cb.sha256(dest))
    meta.update(fingerprint=manifest['fingerprint'],unit=unit,files=files,execution_origin=origin,
                extension_execution_hardware=hardware_record(),
                embedding=None,imported_execution=True,source_elapsed_seconds=meta['elapsed_seconds'],
                source_actual_solves=meta['actual_solves'],elapsed_seconds=0.,actual_solves=0,
                source_data=manifest['source_data'])
    atomic_json(str(stem)+'.json',meta)


def reuse_callback(directory,manifest,unit):
    native=native_unit(manifest['design'],unit)
    cbmeta=cb.checkpoint(directory,native,manifest['fingerprint'],verify=True)
    if cbmeta is None: raise RuntimeError('native execution required before embedded control')
    with np.load(directory/'units'/(native['instance_id']+'.npz')) as saved:
        arrays={key:saved[key] for key in (*FLOAT_FIELDS,'permutations')}
    def results(draw,ie,epsilon,geo,optima):
        result={}
        for ip,p in enumerate((1,2)):
            for arm in INVARIANT[p]:
                ia=cb.ARMS[p].index(arm)
                item={field:float(arrays[field][ip,ia,ie,draw]) for field in FLOAT_FIELDS}
                item.update(permutation=arrays['permutations'][ip,ia,ie,draw],source_arm='native:'+arm,
                            reuse_reason='paired_isometry_identical_costs_and_equivalent_public_span_report_law')
                result[p,arm]=item
        return result
    return results


def ensure_unit(directory,manifest,unit):
    if cb.checkpoint(directory,unit,manifest['fingerprint'],verify=True) is None:
        if unit['embedded']:
            ensure_unit(directory,manifest,native_unit(manifest['design'],unit))
        if unit['imported_original'] and manifest['design']['phase']=='production':
            import_circles(directory,manifest,unit)
        else:
            prepared,extra=prepared_instance(manifest,unit)
            cb.run_unit(directory,manifest,unit,prepared_instance=prepared,seed_namespace=NAMESPACE,
                        extra_metadata=extra,reused_draw=reuse_callback(directory,manifest,unit) if unit['embedded'] else None)
    meta=json.loads((directory/'units'/(unit['instance_id']+'.json')).read_text())
    raw=directory/'units'/(unit['instance_id']+'.csv.gz')
    if raw.name not in meta['files']:
        from .clustering_extension_report import write_unit_csv
        write_unit_csv(directory,manifest,meta,raw)
        meta['files'][raw.name]=dict(bytes=raw.stat().st_size,sha256=cb.sha256(raw))
        atomic_json(directory/'units'/(unit['instance_id']+'.json'),meta)


def work_groups(cfg):
    # A native and its embedded control have one owner, avoiding native writers
    # racing between shards. Native Iris belongs to two panel views, one execution.
    units=cb.all_units(cfg)
    return [[u]+[v for v in units if v['embedded'] and v['native_setting_id']==u['setting_id'] and
                 (v['parent'],v['partition'])==(u['parent'],u['partition'])]
            for u in units if not u['embedded']]


def prepare(root,sources,pilot):
    directory,manifest=resolve(root,sources,pilot)
    directory.mkdir(parents=True,exist_ok=True);(directory/'units').mkdir(exist_ok=True)
    if (directory/'manifest.json').exists() and json.loads((directory/'manifest.json').read_text())!=manifest:
        raise RuntimeError('extension manifest conflict')
    atomic_json(directory/'manifest.json',manifest)
    atomic_json(directory/'expected_units.json',cb.all_units(manifest['design']))
    distributions=sorted(importlib.metadata.distributions(),key=lambda d:d.metadata['Name'].lower())
    (directory/'requirements.freeze.txt').write_text(''.join(f'{d.metadata["Name"]}=={d.version}\n' for d in distributions))
    (directory/'numpy_build_configuration.txt').write_text(str(np.__config__.CONFIG))
    return directory,manifest


def run_study(root,sources,pilot=False,stage='all',shard_index=None,shard_count=None,directory=None):
    refuse_login_compute('run the clustering benchmark extension')
    if stage in ('prepare','all'):
        directory,manifest=prepare(root,sources,pilot)
    else:
        directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    print('Resolved clustering extension results:',directory,flush=True)
    if stage in ('shard','finalize'):
        verify_runtime(manifest)
    stages=directory/'stage_records';stages.mkdir(exist_ok=True)
    record=hardware_record()
    atomic_json(stages/(f'{stage}_{record["slurm_job_id"]}_{shard_index}.json'),
                dict(stage=stage,hardware=record,frozen_environment_matches=cb.environment()==manifest['versions']))
    if stage in ('all','shard'):
        verify_runtime(manifest)
        index=0 if shard_index is None else shard_index;count=1 if shard_count is None else shard_count
        if not 0<=index<count: raise ValueError('invalid extension shard')
        for group in work_groups(manifest['design'])[index::count]:
            for unit in group: ensure_unit(directory,manifest,unit)
    if stage in ('all','finalize'):
        from .clustering_extension_report import finalize
        finalize(directory)
    return directory
