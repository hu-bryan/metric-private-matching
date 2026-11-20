"""Render compact manuscript bars from the audited means and intervals.

Run on a CRC compute node. Arguments: saved summary CSV, output directory.
"""
from pathlib import Path
import hashlib
import json
import socket
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


source, output = map(Path, sys.argv[1:3])
output.mkdir(exist_ok=True, parents=True)
frame = pd.read_csv(source)
arms = ['gumbel', 'original_points', 'projected_representation', 'distance_vectors']
labels = ['Gumbel', 'Original points', 'Projected representation', 'Distance vectors']
colors = ['#4477AA', '#228833', '#AA3377', '#AA9922']
datasets = ['graves/ring_noisy', 'wut/circles', 'fcps/chainlink', 'other/iris', 'uci/banknote',
            'uci/ecoli', 'uci/yeast', 'uci/glass', 'uci/wine', 'uci/statlog', 'uci/wdbc',
            'uci/ionosphere', 'uci/sonar', 'sipu/worms_64', 'mnist/digits10k', 'mnist/fashion10k']
names = ['Ring noisy', 'Circles', 'Chainlink', 'Iris', 'Banknote', 'Ecoli', 'Yeast', 'Glass',
         'Wine', 'Statlog', 'WDBC', 'Ionosphere', 'Sonar', 'Worms-64', 'MNIST', 'Fashion-MNIST']
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
    'mathtext.fontset': 'stix', 'font.size': 8.5, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.2, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.7,
    'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42})
low = np.array(matplotlib.colors.to_rgb('#dcebf9'))
high = np.array(matplotlib.colors.to_rgb('#f6d5d5'))
middle = (low + high)/2
fig, axes = plt.subplots(2, 1, figsize=(5.5, 3.8), sharex=True)
fig.subplots_adjust(left=.08, right=.99, bottom=.24, top=.86, hspace=.13)
for p, ax in zip((1, 2), axes):
    selected = frame[(frame.panel == 'epsilon') & (frame.objective_p == p) & (frame.epsilon == 8)]
    assert len(selected) == 64 and set(selected.n) == {64}
    for j, dataset in enumerate(datasets):
        row = selected[selected.dataset == dataset].iloc[0]
        region = low if row.d <= np.log(64) else high if row.d >= 64 else middle
        ax.axvspan(j-.5, j+.5, color=region, zorder=0)
        ax.hlines(row.random_minus_optimum_mean, j-.43, j+.43,
                  color='.42', ls=':', lw=1.1, zorder=2)
    for i, (arm, color) in enumerate(zip(arms, colors)):
        part = selected[selected.mechanism == arm].set_index('dataset').loc[datasets]
        mean = part.mean_total_regret_mean.to_numpy()
        err = np.stack([mean-part.mean_total_regret_low, part.mean_total_regret_high-mean])
        assert (err >= 0).all()
        ax.bar(np.arange(16)+(i-1.5)*.18, mean, width=.18, color=color, yerr=err,
               error_kw={'elinewidth': .55, 'ecolor': '.2', 'capsize': .8}, zorder=3)
    ax.set_xlim(-.5, 15.5)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r'$\mathbb{E}[\mathrm{err}]$')
axes[0].tick_params(axis='x', bottom=False, labelbottom=False)
axes[1].set_xticks(range(16), names, rotation=70, ha='right', fontsize=7.2)
handles = [Patch(color=color, label=label) for color, label in zip(colors, labels)]
handles.append(Line2D([], [], color='.42', ls=':', label='RAND'))
fig.legend(handles=handles, loc='upper center', ncol=3, columnspacing=1.8,
           bbox_to_anchor=(.5, 1.015))
paths = []
for ext in ('pdf', 'png'):
    path = output/f'numerical_bars_compact.{ext}'
    fig.savefig(path, dpi=230, bbox_inches='tight', pad_inches=.035)
    paths.append(path)
plt.close(fig)
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
provenance = dict(source_summary_sha256=digest(source), renderer_sha256=digest(Path(__file__)),
    epsilon=8, n_per_side=64, bar_count=128, unchanged_means_and_confidence_intervals=True,
    shared_legend=True, shared_dataset_labels=True,
    figures={path.name: digest(path) for path in paths})
(output/'numerical_bars_compact_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
print(json.dumps(provenance, indent=2))
