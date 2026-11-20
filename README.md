# Matching Under Metric Privacy

Reusable Python mechanisms and reproducible experiments for **Matching Under
Metric Privacy**, by Hongyi B. Hu, Dung Nguyen, and M. Amin Rahimian.

The library matches `n` private points to `n` fixed public points in Euclidean
space. The experiment commands reproduce the sixteen benchmark comparisons,
Fashion-MNIST dimension sweep, Banknote sample-size sweep, and Sonar rank-energy
experiment separately. No cluster, Slurm installation, or GPU is required.

## Install

Python 3.11 or newer:

```bash
git clone https://github.com/hu-bryan/metric-private-matching.git
cd metric-private-matching
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[reproduce]"
metric-private-matching verify
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell instead.
`python -m metric_private_matching` is equivalent to `metric-private-matching`.
For just the mechanisms, install with `python -m pip install .`; the library
depends only on NumPy and SciPy. Experiment and plotting dependencies are optional.

`requirements-paper.txt` records the original Linux/CPython 3.12.13 environment.
For the closest numerical replay, use Python 3.12 and
`python -m pip install -r requirements-paper.txt`, then install this package.
The ordinary installation supports newer compatible environments without
requiring identical floating-point output. Original versions, configuration,
sampling artifacts, and scientific source hashes are preserved in
`src/metric_private_matching/reproduction/provenance/`.

## Use your own inputs

```python
import numpy as np
from metric_private_matching import match

data_rng = np.random.default_rng(7)
private = data_rng.normal(size=(50, 8))
public = data_rng.normal(size=(50, 8))

permutation = match(
    private, public, epsilon=4.0,
    mechanism="gumbel",       # or one of the three local mechanisms below
    cost="euclidean",        # or "squared"
    rng=42,                  # research seed; omit for fresh randomness
)
paired_public = public[permutation]
```

Both inputs are finite `(n, d)` arrays. The return value is a permutation:
private row `i` is paired with public row `permutation[i]`. Inputs are not
modified. The library does not normalize your data automatically.

| `mechanism` | What is privatized | Trust model |
|---|---|---|
| `original_points` | Original coordinates with radial Laplace noise | Local |
| `projected_representation` | Representation using the public affine span | Local |
| `distance_vectors` | Public-candidate cost vectors with infinity-norm Laplace noise | Local |
| `gumbel` | Assignment computed with independent Gumbel edge scores | Central; trusted server |

For every mechanism, privacy uses
`D(X, X') = sum_i ||X[i] - X'[i]||_2`, including when the objective is squared
Euclidean cost. The public candidates, input dimensions, and row roster are
fixed and public. `epsilon` is in inverse units of Euclidean displacement.

The mechanisms are also individually importable as `original_points`,
`projected_representation`, `distance_vectors`, and `gumbel_matching`.
To separate local randomization from assignment:

```python
from metric_private_matching import (
    privatize_points, matching_from_points,
    privatize_projected, privatize_distances, matching_from_costs,
)

reports = privatize_points(private, epsilon=4, rng=42)
permutation = matching_from_points(reports, public)

projected = privatize_projected(private, public, epsilon=4, cost="squared", rng=42)
permutation = projected.matching()

cost_reports = privatize_distances(private, public, epsilon=4, rng=42)
permutation = matching_from_costs(cost_reports)
```

The projected report remembers its objective and rejects an incompatible
decoder objective. Squared-cost
reports can omit private row constants; their entries need not be nonnegative
or equal the original squared distances. Gumbel's calibrated guarantee applies
to its released assignment; its internal perturbed scores are not exposed.

Fixed/public seeds are for reproducible experiments. A deployment must use fresh
secret randomness and account for repeated releases. Preprocessing fitted on
private data needs its own privacy analysis; the API operates on the supplied
coordinates with preprocessing fixed. Examples are in `examples/`.

## Inspect the published numbers without rerunning experiments

The repository includes the final tidy CSV, original aggregate CSVs, and saved
parent means/draws. These commands recompute the summary from those saved results
and render plots using **only that CSV**:

```bash
metric-private-matching saved-results --output outputs/paper
metric-private-matching plot --csv outputs/paper/summary.csv --output outputs/paper/figures
metric-private-matching plot --csv outputs/paper/summary.csv --view bars --output outputs/paper/bars
```

`results/summary.csv` is the ready-to-read tidy copy. To export just one
experiment, add `--experiment benchmarks`, `dimension-sweep`, `size-sweep`, or
`sonar` to `saved-results`.

In the tidy CSV, `mean`, `ci_low`, and `ci_high` describe total excess cost,
except for Sonar rows (`metric=mean_pvalue`), which describe p-values. Size-sweep
rows also include `mean_per_pair` and its confidence limits. Each row identifies
the dataset, objective, mechanism, budget, sample size, dimension, and repetition
hierarchy. Confidence limits are pointwise 95% Student-t intervals.

To rebuild the eight figures and appendix numerical tables with the original
paper renderers:

```bash
metric-private-matching paper-figures --output outputs/original-figures
```

This command uses saved results and fetches only the small Sonar sources needed
for calibration. It runs no private mechanisms. Its comparison report preserves
one known prose rounding discrepancy: Ecoli's squared-cost Gumbel mean at
epsilon 8 is 36.534614267231305, which directly rounds to **36.53**, while the
manuscript says **36.54**. The three-decimal appendix value **36.535** agrees.
The source data are not changed to force agreement with that prose value.

## Rerun an experiment

Each experiment has its own command. `--mode quick` is the default and runs a
small demonstration. `--mode paper` explicitly selects the original production
sizes. Inspect either configuration before computing by adding `--plan`; this
does not download data or execute an experiment.

### One benchmark or all sixteen

```bash
# Quick end-to-end example, both objectives, one dataset:
metric-private-matching benchmarks --dataset other/iris --mode quick --workers 1 --output outputs/iris-quick

# Full paper comparison on one dataset:
metric-private-matching benchmarks --dataset uci/wine --mode paper --output outputs/wine-paper

# Full sixteen-dataset comparison, both objectives:
metric-private-matching benchmarks --dataset all --mode paper --output outputs/benchmarks-paper

# Run just one objective if desired:
metric-private-matching benchmarks --dataset all --objective euclidean --mode paper --output outputs/benchmarks-euclidean
```

Use `metric-private-matching datasets` to list the exact dataset identifiers.
The paper configuration uses `n=64`, epsilon `0.25, 0.50, ..., 16`, twenty parent
collections, five disjoint public/private partitions per parent, and fifty noise
draws per instance. Fixed-budget bars use the epsilon-8 rows of this sweep.

### Fashion-MNIST dimension sweep

```bash
metric-private-matching dimension-sweep --mode quick --output outputs/dimension-quick
metric-private-matching dimension-sweep --mode paper --output outputs/dimension-paper
```

The paper configuration uses `n=800` per side, epsilon 16, all 56 measured
dimensions, twenty parent collections, five random projection chains, five
partitions, and fifty noise draws. Nested Haar projection prefixes and RMS
normalization follow the original experiment. The quick profile uses the same
input construction at dimensions 4, 24, and 104 with fewer repetitions.

### Banknote sample-size sweep

```bash
metric-private-matching size-sweep --mode quick --output outputs/size-quick
metric-private-matching size-sweep --mode paper --output outputs/size-paper
```

The paper configuration uses `d=4`, epsilon 8, `n=20,40,...,200`, twenty parents,
five partitions, and fifty noise draws. Feature weights stay fixed within a
parent; each selected cloud is centered and RMS-normalized. Plots show excess
cost per pair, while the CSV also retains total excess cost.

### Sonar rank-energy experiment

```bash
metric-private-matching sonar --mode quick --output outputs/sonar-quick
metric-private-matching sonar --mode paper --output outputs/sonar-paper
```

The paper configuration uses all 208 observations in 60 dimensions, the fixed
Halton reference grid, epsilon `1,2,4,8,16`, 1,000 mechanism repetitions, and
9,999 null draws. The original public null sample is included. Add
`--regenerate-null` to regenerate it with its original seed instead.

### Results and figures for your rerun

Every completed command writes `summary.csv` and `summary_metadata.json` to the
requested output directory, plus per-study configuration, environment records,
checkpoints, and intermediate aggregates. Render the new results with:

```bash
metric-private-matching plot --csv outputs/iris-quick/summary.csv --output outputs/iris-quick/figures
```

There is deliberately no command that silently reruns every experiment in the
paper. The paper commands above identify each workload explicitly.

## Resources, resuming, and reproducibility

- `--workers auto` considers available CPUs, scheduler/container limits, and
  available memory; it caps automatic concurrency at four processes and uses
  one process when memory cannot be detected. Set `--workers 1` for sequential
  execution, or an explicit count suitable for your device.
- Each worker uses one BLAS/OpenMP thread. Adding workers does not change task
  seeds, sampled rows, or the order of statistical averaging. The full
  experiments can take substantial CPU time on a laptop. `--plan` reports task
  counts and settings before a run.
- Outputs are checkpointed. Repeat the same command to resume. A changed mode,
  dataset selection, or configuration requires a fresh output directory; a
  manifest mismatch is an error. Quick and paper results are never pooled.
- Full sweeps can create many intermediate assignment files. Choose an output
  directory on a disk with room. On a cluster, use your own scratch/project disk.
  The code embeds no requirement for the authors' filesystem or scheduler.
- Raw downloads go to `$METRIC_PRIVATE_MATCHING_DATA`, or
  `$XDG_CACHE_HOME/metric-private-matching/data` (normally
  `~/.cache/metric-private-matching/data`). Set the variable to a larger disk
  before running the Fashion or Sonar commands if home storage is limited.
- Benchmark and size commands use the included exact sampled input artifacts.
  Fashion and Sonar fetch their required source files automatically. To fetch
  every original raw source for inspection, use `metric-private-matching
  fetch-data`. Downloads are pinned to archived SHA256s and fail on changed
  upstream bytes. Dataset URLs, attribution, and notices are bundled separately
  from the project code; see the [dataset sources and notices](src/metric_private_matching/reproduction/data/DATASETS.md).
  `metric-private-matching check-inputs --output outputs/input-checks` verifies
  sampled memberships, disjoint partitions, transforms, and reconstructed arrays
  against all sixteen raw sources without running private mechanisms.

For externally scheduled shards, prepare once, run disjoint shards, then
aggregate once. For example:

```bash
metric-private-matching dimension-sweep --mode paper --output outputs/dimension-paper --stage prepare
metric-private-matching dimension-sweep --mode paper --output outputs/dimension-paper --stage run --shard-index 0 --shard-count 2
metric-private-matching dimension-sweep --mode paper --output outputs/dimension-paper --stage run --shard-index 1 --shard-count 2
metric-private-matching dimension-sweep --mode paper --output outputs/dimension-paper --stage aggregate
```

Each shard may use `--workers`. Do not run two writers for the same shard/task.

Paper reruns use the original keyed noise seeds, saved memberships, fitted
feature transforms, and frozen scientific implementations. Fashion inputs are
reconstructed from source rows and compact metadata and checked against the
original array hashes. The frozen code is an archival reference; normal users
should import the public API above. The adapter changes scheduling, paths, and
hardware reporting, while calling the original numerical functions.
See [artifact and seed details](src/metric_private_matching/reproduction/ARTIFACTS_AND_SEEDS.md)
for the exact namespace and key structure of each random stream.

Noise is averaged before partitions, then projection chains where applicable,
then independent parents. Confidence intervals use the twenty parent means;
Sonar uses its 1,000 independent repetitions. Unit noise is shared across
epsilon, and mathematically identical point-method endpoints reuse reports.
Cross-platform QR/SVD, solver ties, and floating-point differences can change
individual assignments; identical decimal output across BLAS/OS versions is
not promised. Saved results remain available for checking the reported values.

## Tests

```bash
python -m pip install -e ".[reproduce,dev]"
python -m pytest
```

Tests cover calibration against the frozen implementations, input validation,
rank-deficient public sets, reproducible sequential/parallel runs, and the
experiment/CSV workflow. See `VALIDATION.md` for the checks executed for this
release and their limits.

## License

A project-code license has not yet been selected for this initial publication.
Dataset terms and attributions are recorded separately in the bundled notices.
