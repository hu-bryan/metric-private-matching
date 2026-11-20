"""Portable paths and execution profile; no numerical imports at module load."""
from pathlib import Path
import hashlib
import importlib
import importlib.util
import importlib.metadata
import json
import os
import platform
import socket
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('METRIC_PRIVATE_MATCHING_DATA',
    str(Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'metric-private-matching'/'data'))).expanduser().resolve()
STUDIES = ('euclidean', 'squared', 'fashion', 'sonar')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def configure(study=None):
    # Set these before NumPy/BLAS imports, including in spawned workers.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'PRIVATE_MATCHING_NUM_THREADS'):
        os.environ[name] = '1'
    if study == 'sonar' and platform.machine().lower() in ('x86_64', 'amd64'):
        os.environ['OPENBLAS_CORETYPE'] = 'NEHALEM'
        os.environ['NPY_DISABLE_CPU_FEATURES'] = 'X86_V3,X86_V4,AVX512_ICL,AVX512_SPR'
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('PM_NORM_CONSTRAINT', 'portable')
    os.environ.setdefault('PM_MANUSCRIPT_CONSTRAINT', 'portable')
    # Windows has no Unix resource module. The portable adapter does not use its
    # measurements; keep dormant archived reporting imports possible there.
    if 'resource' not in sys.modules and importlib.util.find_spec('resource') is None:
        resource = types.ModuleType('resource')
        resource.RUSAGE_SELF = 0
        def unavailable(*args):
            raise NotImplementedError('Unix resource measurements are unavailable on this platform')
        resource.getrusage = unavailable
        sys.modules['resource'] = resource
    # A laptop requires no scheduler. CRC login hosts still require compute allocation.
    if socket.gethostname().startswith('login') and '.crc.pitt.edu' in socket.getfqdn():
        raise RuntimeError('On Pitt CRC run numerical commands inside a compute allocation.')


def frozen(study):
    location = ROOT / 'frozen' / study
    sys.path.insert(0, str(location))
    modules = dict(euclidean='clustering_manuscript', squared='clustering_squared',
                   fashion='fashion_dimension', sonar='sonar_rank')
    result = importlib.import_module('private_matching.' + modules[study])
    # Instrumentation is separate from the unchanged numerical implementations.
    if platform.system() != 'Linux':
        def hardware():
            return dict(host=platform.node(), cpu_model=platform.processor() or None,
                        constraint='portable', job_id=os.environ.get('SLURM_JOB_ID'))
        for name in ('private_matching.clustering_normalized', 'private_matching.sonar_rank'):
            if name in sys.modules:
                sys.modules[name].hardware = hardware
    return result


def environment():
    packages = {}
    for name in ('numpy', 'scipy', 'pandas', 'matplotlib', 'PyYAML', 'joblib'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(),
                machine=platform.machine(), packages=packages,
                thread_environment={k: os.environ.get(k) for k in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NPY_DISABLE_CPU_FEATURES', 'OPENBLAS_CORETYPE')})


def verify_bundle():
    manifest = read_json(ROOT / 'ASSET_SHA256.json')
    failures = []
    for name, expected in manifest.items():
        path = ROOT / name
        if not path.is_file() or digest(path) != expected:
            failures.append(name)
    if failures:
        raise RuntimeError('Missing or changed package files: ' + ', '.join(failures[:20]))
    return dict(status='passed', checked_files=len(manifest))
