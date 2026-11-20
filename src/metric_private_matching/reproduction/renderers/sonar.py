from pathlib import Path
import csv
import socket
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


import sys
root = Path(sys.argv[1])
with (root / 'sonar_mean_pvalue.csv').open() as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 20
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
    'mathtext.fontset': 'stix', 'font.size': 11, 'axes.labelsize': 12,
    'legend.fontsize': 10.5, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, ax = plt.subplots(figsize=(6.4, 4))
fig.subplots_adjust(left=.12, right=.975, bottom=.16, top=.80)
for name, color, marker, style in (
    ('Gumbel', '#4477AA', 'o', '-'),
    ('Original points', '#228833', 's', '-'),
    ('Projected representation', '#AA3377', '^', '--'),
    ('Distance vectors', '#CCBB44', 'D', '-.')):
    part = sorted((r for r in rows if r['mechanism'] == name), key=lambda r: float(r['epsilon']))
    x = np.array([float(r['epsilon']) for r in part])
    y = np.array([float(r['mean_p']) for r in part])
    low = np.array([float(r['mean_p_low_95']) for r in part])
    high = np.array([float(r['mean_p_high_95']) for r in part])
    assert np.array_equal(x, [1, 2, 4, 8, 16])
    assert np.all(low <= y) and np.all(y <= high)
    ax.errorbar(x, y, yerr=np.array([y-low, high-y]), label=name,
        color=color, marker=marker, ls=style, lw=1.5, ms=4,
        capsize=2, elinewidth=.7)
baseline = {float(r['nonprivate_p']) for r in rows}
assert baseline == {.0022}
ax.axhline(.0022, color='black', ls='--', lw=1.2)
ax.text(1.02, .035, r'Non-private: $p=0.0022$', fontsize=10.5)
ax.set_xscale('log', base=2)
ax.set_xlim(.92, 17.5)
ax.set_ylim(-.012, .55)
ax.set_xticks([1, 2, 4, 8, 16], ['1', '2', '4', '8', '16'])
ax.set_yticks(np.arange(0, .6, .1))
ax.set_xlabel(r'$\varepsilon$')
ax.set_ylabel(r'Mean $p$-value')
ax.grid(axis='y', color='.9', lw=.6)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.04), ncol=2, borderaxespad=0)
for extension in ('pdf', 'png'):
    fig.savefig(root / ('sonar_mean_pvalue.' + extension), dpi=300,
        metadata={'CreationDate': None, 'ModDate': None} if extension == 'pdf' else None)
plt.close(fig)
print('Sonar means and confidence intervals retained from the existing 20-row CSV')
