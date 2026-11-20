"""Restore only missing raw inputs, using immutable URLs and archived SHA256s."""
from pathlib import Path
import io
import urllib.request
import zipfile
import argparse
from common import ROOT, DATA, digest, read_json


def fetch(dataset='all'):
    records = read_json(ROOT/'data/sources.json')
    if dataset != 'all':
        records = [record for record in records if
                   record['dataset'] == dataset or
                   (dataset == 'uci/sonar' and '/sonar/' in record['path'])]
        if not records:
            raise ValueError(f'Unknown source dataset: {dataset}')
    for record in records:
        path = DATA/Path(record['path']).relative_to('data')
        if path.exists():
            if digest(path) != record['sha256']:
                raise RuntimeError(f'Existing data has changed; refusing to overwrite: {path}')
            continue
        print('Downloading', record['path'], flush=True)
        request = urllib.request.Request(record['url'], headers={'User-Agent': 'research-reproduction/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response:
            content = response.read()
        if record.get('zip_member'):
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                content = archive.read(record['zip_member'])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.download')
        temporary.write_bytes(content)
        if digest(temporary) != record['sha256']:
            temporary.unlink()
            raise RuntimeError(f'Upstream bytes differ from the frozen input: {record["url"]}')
        temporary.replace(path)
    print(f'All {len(records)} raw source files match their archived checksums.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='all')
    fetch(parser.parse_args().dataset)
