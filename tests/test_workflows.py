import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from metric_private_matching import execution


def command(*args):
    return subprocess.run([sys.executable, '-m', 'metric_private_matching', *map(str,args)],
                          check=True, capture_output=True, text=True)


def test_worker_defaults(monkeypatch):
    monkeypatch.setattr(execution, 'available_cpus', lambda: 128)
    monkeypatch.setattr(execution, 'available_memory', lambda: 1 << 40)
    assert execution.workers() == 4
    monkeypatch.setattr(execution, 'available_memory', lambda: None)
    assert execution.workers() == 1
    assert execution.workers('1') == 1
    with pytest.raises(ValueError):
        execution.workers('0')


def test_assets():
    assert 'passed' in command('verify').stdout
    assert len(command('datasets').stdout.strip().splitlines()) == 16


@pytest.mark.parametrize('name,count', [('dimension-sweep',28000),('size-sweep',1000),('sonar',1000)])
def test_paper_workload_plan(name, count, tmp_path):
    result = command(name, '--mode','paper','--output',tmp_path/'plan','--plan')
    plan = json.loads(result.stdout)
    assert plan['tasks'] == count
    assert not plan['experiments_executed']
    assert not (tmp_path/'plan').exists()


def test_sequential_parallel_equal(tmp_path):
    for workers in (1,2):
        command('benchmarks','--dataset','other/iris','--objective','euclidean',
                '--mode','quick','--workers',workers,'--output',tmp_path/str(workers))
    first = pd.read_csv(tmp_path/'1/summary.csv')
    second = pd.read_csv(tmp_path/'2/summary.csv')
    pd.testing.assert_frame_equal(first,second)
    for one in (tmp_path/'1/euclidean/units').glob('*.npz'):
        with np.load(one) as a, np.load(tmp_path/'2/euclidean/units'/one.name) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key],b[key])
    # Resume checks the same task fingerprints instead of pooling new draws.
    command('benchmarks','--dataset','other/iris','--objective','euclidean',
            '--mode','quick','--workers',1,'--output',tmp_path/'1')
    pd.testing.assert_frame_equal(first,pd.read_csv(tmp_path/'1/summary.csv'))
    with pytest.raises(subprocess.CalledProcessError):
        command('benchmarks','--dataset','other/iris','--objective','euclidean',
                '--mode','paper','--workers',1,'--output',tmp_path/'1')


def test_csv_saved_results_and_plots(tmp_path):
    command('saved-results','--output',tmp_path/'paper')
    frame = pd.read_csv(tmp_path/'paper/summary.csv')
    assert set(frame.experiment) == {'benchmarks','dimension-sweep','size-sweep','sonar'}
    assert len(frame[frame.experiment == 'benchmarks']) == 16*64*4*2
    command('plot','--csv',tmp_path/'paper/summary.csv','--output',tmp_path/'figures')
    provenance = json.loads((tmp_path/'figures/plot_provenance.json').read_text())
    assert provenance['csv_only'] and provenance['mechanism_runs'] == 0
    assert len(list((tmp_path/'figures').glob('*.png'))) == 5
