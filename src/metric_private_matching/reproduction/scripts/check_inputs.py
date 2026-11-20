"""Check raw-to-sampled input provenance without running a private mechanism."""
from pathlib import Path
import argparse
import json
from common import ROOT, DATA, configure, frozen, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configure()
    import numpy as np
    import pandas as pd
    module = frozen('euclidean')
    from private_matching.clustering_manuscript_report import audit_master
    from input_artifacts import fashion_arrays
    manifest = read_json(ROOT/'provenance/euclidean/manifest.json')
    registry = read_json(ROOT/'inputs/euclidean/master_registry.json')
    fashion = read_json(ROOT/'inputs/fashion/master_registry.json')
    expected = module.DATASETS
    checks, rows, datasets = [], [], {}
    for dataset in sorted(manifest['sources']):
        if dataset == 'uci/banknote':
            data = np.loadtxt(DATA/'raw/uci/data_banknote_authentication.txt', delimiter=',')
            data = data[:, :4]
        else:
            data = np.loadtxt(DATA/'raw'/(dataset+'.data.gz'))
        assert data.shape == tuple(expected[dataset]), (dataset, data.shape)
        datasets[dataset] = data
        rows.append(dict(dataset=dataset, available_records=data.shape[0], dimension=data.shape[1]))
        for record in registry.values():
            if record['dataset'] != dataset:
                continue
            with np.load(ROOT/'inputs/euclidean'/record['file']) as z:
                ids, orders = z['source_ids'], z['orders']
                np.testing.assert_array_equal(z['raw'], data[ids])
                assert len(np.unique(ids)) == len(ids)
                assert orders.shape[0] == 5
                for order in orders:
                    np.testing.assert_array_equal(np.sort(order), np.arange(len(ids)))
            checks.append(dict(master=record['file'], raw_rows_match=True, disjoint_partition_sides=True))
            audit_master(ROOT/'inputs/euclidean'/record['file'], dataset)
    for parent, record in fashion.items():
        z = fashion_arrays(parent, source=datasets['mnist/fashion10k'])
        ids = z['source_ids']
        assert len(ids) == 1600 and len(np.unique(ids)) == 1600
        np.testing.assert_array_equal(z['raw'], datasets['mnist/fashion10k'][ids])
        values = np.ascontiguousarray(((z['raw']-z['feature_offset'])/z['feature_scale'])[:,z['pixel_order']])
        center = values.mean(axis=0)
        radius = np.sqrt(np.mean(np.sum((values-center)**2,axis=1)))
        np.testing.assert_allclose(z['Z'], (values-center)/radius, rtol=1e-13, atol=1e-13)
        for order in z['orders']:
            np.testing.assert_array_equal(np.sort(order), np.arange(1600))
        checks.append(dict(master='fashion/'+record['file'], raw_rows_match=True,
                           every_original_array_hash_matches=True, disjoint_partition_sides=True))
    sonar = np.loadtxt(DATA/'sonar/sonar.data.gz')
    np.testing.assert_array_equal(sonar, datasets['uci/sonar'])
    with np.load(ROOT/'inputs/sonar/inputs.npz') as z:
        np.testing.assert_array_equal(z['R'], sonar)
        assert z['Q'].shape == (208,60) and int(z['metal'].sum()) == 111
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output/'dataset_inventory.csv', index=False)
    report = dict(status='passed', dataset_count=len(rows), sampled_masters=len(checks),
                  raw_rows_and_partitions_checked=True, experiments_executed=False, masters=checks)
    write_json(args.output/'input_checks.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'masters'},indent=2))


if __name__ == '__main__':
    main()
