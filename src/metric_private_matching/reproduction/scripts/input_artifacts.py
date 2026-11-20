"""Losslessly reconstruct Fashion masters from exact source rows and metadata.

No sampling, refitting, or random draw occurs here. Normalized arrays use the
archived offsets, scales, center, and radius. Every array is checked against its
original dtype, shape, and SHA256 before it can be used by the frozen runners.
"""
from pathlib import Path
from functools import lru_cache
import argparse
import copy
import hashlib

from common import ROOT, DATA, configure, digest, read_json, write_json


def array_digest(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


@lru_cache(maxsize=1)
def fashion_source():
    import numpy as np
    return np.loadtxt(DATA/'raw/mnist/fashion10k.data.gz')


def fashion_arrays(parent, source=None):
    import numpy as np
    spec = read_json(ROOT/'inputs/fashion/compact_masters.json')[str(parent)]
    path = ROOT/'inputs/fashion'/spec['metadata_file']
    if digest(path) != spec['metadata_sha256']:
        raise RuntimeError(f'Changed Fashion metadata: {path}')
    with np.load(path) as z:
        arrays = {name: z[name] for name in z.files}
    data = fashion_source() if source is None else source
    arrays['raw'] = np.asarray(data[arrays['source_ids']], dtype=spec['arrays']['raw']['dtype'])
    features = (arrays['raw']-arrays['feature_offset'])/arrays['feature_scale']
    features = np.ascontiguousarray(features[:, arrays['pixel_order']])
    arrays['Z'] = np.asarray((features-arrays['original_center'])/arrays['original_radius'],
                             dtype=spec['arrays']['Z']['dtype'])
    for name, expected in spec['arrays'].items():
        actual = arrays[name]
        if (list(actual.shape) != expected['shape'] or actual.dtype.str != expected['dtype']
                or array_digest(actual) != expected['sha256']):
            raise RuntimeError(f'Fashion reconstruction differs from archived array: parent {parent}, {name}')
    return {name: arrays[name] for name in spec['key_order']}


def materialize_fashion(output, parents=None):
    import numpy as np
    output = Path(output)
    registry = copy.deepcopy(read_json(ROOT/'inputs/fashion/master_registry.json'))
    if parents is not None:
        registry = {k: v for k, v in registry.items() if int(k) in parents}
    for parent, record in registry.items():
        arrays = fashion_arrays(parent)
        target = output/record['file']
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(target)
        record['original_container_sha256'] = record['sha256']
        record['sha256'] = digest(target)
    write_json(output/'master_registry.json', registry)
    return registry


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configure()
    materialize_fashion(args.output)
    print('Restored all twenty Fashion masters; every original array hash matches.')
