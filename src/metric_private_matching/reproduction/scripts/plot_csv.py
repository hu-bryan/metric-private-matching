"""Render plots using only the tidy CSV exported by summarize.py."""
from pathlib import Path
import argparse
import hashlib
import json


def render(source, output, view='curves'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    frame = pd.read_csv(source)
    required = {'experiment', 'dataset', 'objective_p', 'mechanism', 'epsilon', 'n', 'd',
                'mean', 'ci_low', 'ci_high', 'random_reference', 'nonprivate_reference', 'mode'}
    missing = required - set(frame)
    if missing:
        raise ValueError(f'CSV is missing columns: {sorted(missing)}')
    keys = ['experiment', 'dataset', 'objective_p', 'mechanism', 'epsilon', 'n', 'd']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate settings in CSV; keep quick, paper, and independent runs separate')
    arms = ['original_points', 'projected_representation', 'distance_vectors', 'gumbel']
    colors = ['#228833', '#AA3377', '#AA9922', '#4477AA']
    labels = ['Original points', 'Projected representation', 'Distance vectors', 'Gumbel']
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for (experiment, objective), group in frame.groupby(['experiment', 'objective_p'], sort=True):
        datasets = list(group.dataset.unique())
        if view == 'bars' and experiment != 'benchmarks':
            continue
        ncols = min(4, len(datasets))
        nrows = (len(datasets)+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.2*nrows), squeeze=False)
        for ax, dataset in zip(axes.flat, datasets):
            data = group[group.dataset == dataset]
            if view == 'bars':
                data = data[np.isclose(data.epsilon, 8)]
                if data.empty:
                    raise ValueError('Bar view requires epsilon=8')
                for i, (arm, color, label) in enumerate(zip(arms, colors, labels)):
                    part = data[data.mechanism == arm]
                    if len(part) != 1:
                        raise ValueError('Expected one bar per mechanism/dataset/objective')
                    row = part.iloc[0]
                    ax.bar(i, row['mean'], color=color, label=label,
                           yerr=[[max(0., row['mean']-row.ci_low)], [max(0., row.ci_high-row['mean'])]])
                ax.set_xticks([])
                ax.axhline(float(data.random_reference.iloc[0]), color='.45', ls=':', label='Random matching')
                ax.set_ylabel('Total excess cost')
            else:
                x = 'd' if experiment == 'dimension-sweep' else 'n' if experiment == 'size-sweep' else 'epsilon'
                per_pair = experiment == 'size-sweep'
                mean, low, high = ('mean_per_pair','ci_low_per_pair','ci_high_per_pair') if per_pair else ('mean','ci_low','ci_high')
                for arm, color, label in zip(arms, colors, labels):
                    part = data[data.mechanism == arm].sort_values(x)
                    ax.plot(part[x], part[mean], color=color, label=label,
                            ls='--' if arm == 'projected_representation' else '-', marker='.' if experiment == 'sonar' else None)
                    ax.fill_between(part[x], part[low], part[high], color=color, alpha=.12)
                ref = data[data.mechanism == 'gumbel'].sort_values(x)
                if experiment == 'sonar':
                    ax.axhline(float(ref.nonprivate_reference.iloc[0]), color='.35', ls=':', label='Nonprivate')
                    ax.axhline(.05, color='black', ls='--', lw=.8, label='0.05')
                    ax.set_ylabel('Mean p-value')
                else:
                    divisor = ref.n if per_pair else 1
                    ax.plot(ref[x], ref.random_reference/divisor, color='.45', ls=':', label='Random matching')
                    if experiment in ('dimension-sweep', 'size-sweep'):
                        bound = 2*ref.n*np.minimum(ref.d, np.log(ref.n))/ref.epsilon/divisor
                        ax.plot(ref[x], bound, color='black', ls='-.', label='Combined guarantee')
                    ax.set_ylabel('Excess cost per pair' if per_pair else 'Total excess cost')
                ax.set_xlabel({'d':'Dimension d', 'n':'Points per side n', 'epsilon':'Privacy budget ε'}[x])
            ax.set_title(dataset + (' [quick demonstration]' if set(data['mode']) == {'quick'} else ''))
            ax.spines[['top','right']].set_visible(False)
        for ax in list(axes.flat)[len(datasets):]:
            ax.set_visible(False)
        handles, names = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, names, loc='lower center', ncol=min(4, len(names)), frameon=False)
        fig.suptitle(f'{experiment} · {"Euclidean" if objective == 1 else "squared Euclidean"} objective')
        fig.tight_layout(rect=(0, .13 if len(datasets) == 1 else .05, 1, .95))
        for extension in ('pdf', 'png'):
            path = output/f'{experiment}_p{int(objective)}_{view}.{extension}'
            fig.savefig(path, dpi=180, bbox_inches='tight')
            files.append(path.name)
        plt.close(fig)
    (output/'plot_provenance.json').write_text(json.dumps(dict(
        source_csv_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), figures=files,
        csv_only=True, mechanism_runs=0), indent=2)+'\n')
    print(f'Rendered {len(files)} files from {source}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--view', choices=('curves','bars'), default='curves')
    args = parser.parse_args()
    render(args.csv, args.output, args.view)
