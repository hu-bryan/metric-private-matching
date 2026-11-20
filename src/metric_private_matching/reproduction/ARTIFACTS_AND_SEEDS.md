# Which information is preserved exactly, and which is regenerated?

The rule is to preserve realized information that defines the reported
experiment, and preserve a complete generation recipe for large intermediate
arrays that are reproducible under the reference numerical environment. A seed
without the input, generator, algorithm, versions and seed-key structure is
insufficient.

| Material | In this repository | Reason |
|---|---|---|
| Versioned raw datasets and semantic Sonar labels | Immutable source URLs, notices and SHA256s; fetched into an external cache | Future upstream changes must not alter the inputs. |
| Sampled source row IDs, raw selected records and all partition orders | Exact NPZ masters and registries; Fashion selected pixel rows are restored from exact saved IDs and checksummed source pixels | Some sampling was inherited from earlier studies. A single new global seed cannot reconstruct that lineage. |
| Fitted feature offsets/scales, binary encodings and full-dimensional normalized Fashion masters | Exact transformation metadata; Fashion normalized matrices are restored using the saved center/radius and checked against original array hashes | These determine the metric and normalization used in the paper. No refitting or random draw is needed to restore the full-dimensional masters. Per-setting normalization is recomputed by the frozen functions. |
| Sonar public grid and non-private assignment | Exact saved inputs plus the generation code | They define the baseline and permit checking reconstruction. |
| Sonar's primary 9,999 null draws | Exact allocations and statistics, plus the original seed recipe | The displayed private p-values are conditional on this realized public null sample. Redrawing it with an arbitrary seed changes the reported values. |
| Production parent/instance means and Sonar per-run statistics/p-values | Exact saved values (lossless floating-point CSV representation) | They support fast reconstruction of all published means and confidence intervals without simulation. |
| Paper numerical tables and plotted-value CSVs | Exact reference files and the original figure renderers | They provide the comparison target, including rounding and units. Figures are rendered from CSV rather than bundled as duplicated binaries. |
| Fashion Haar matrices, projected clouds and per-dimension radii | Original seeds, frozen QR implementation and hashes of original arrays | The dense basis archive is about 1.3 GiB; it is an intermediate, not additional data. Regenerate it under the reference environment and check hashes. |
| Mechanism unit noise and private assignments | Original keyed seeds and exact executed functions | No new scientific decision depends on storing these large arrays. Rerun mode regenerates them and saves the resulting assignments. |

The regenerated projection arrays can differ on another numerical stack even
with the same Gaussian draws: QR, matrix multiplication and SVD involve
floating-point algorithms. If exact conditional geometry across machines is
required, exact matrices/projected inputs would be needed in addition to seeds.
This repository instead records projection hash comparisons and reports any numerical
differences in rerun outputs. It does not claim bitwise equality on every machine.

## Seed keys, not one global seed

The frozen helper computes
`int.from_bytes(SHA256(json.dumps(keys)).digest()[:16], 'little')` and supplies it
to NumPy `default_rng` (PCG64 in the reference environment). Python's randomized
`hash()` and task scheduling order are never scientific seed inputs.

| Stream | Original namespace and key structure |
|---|---|
| Euclidean/Banknote unit noise | `clustering-manuscript-20261114-v1`, phase, instance ID, repetition, mechanism channel |
| Squared-cost unit noise | `clustering-public-span-squared-20261114-v1`, phase, instance ID, repetition, mechanism channel |
| Fashion basis | `fashion-normalized-full-haar-dimension-20261114-v1`, phase, parent, chain, `basis` |
| Fashion unit noise | Same Fashion namespace, phase, instance ID, repetition, mechanism channel |
| Sonar streams | `sonar-rank-energy-20261112-v1`; the frozen `rng(*keys)` prepends this namespace. |
| Sonar primary null | Sonar namespace, `production`, `public-bank`, `0` |
| Sonar non-private tie priorities | Sonar namespace, `nonprivate`, `assignment-priorities` |
| Sonar private tie priorities | Sonar namespace, phase, `private`, repetition, original arm name, floating-point epsilon, `assignment-priorities` |

Instance IDs retain the original strings, including setting, parent, chain,
dimension and partition. Original arm names are retained internally even when
the paper uses a different legend label. The precise unit-noise keys for each
Sonar arm are in its unchanged `channels` function.

Unit noise is shared across privacy budgets in the same way as in production.
Mechanism channels are independent, apart from explicitly equivalent endpoint
reuse. The privacy claim concerns each individual execution at its stated
budget; the experimental coupling is not a proposed repeated-release protocol.

## Fresh runs versus exact saved artifacts

`saved-results` recomputes statistics from archived production values.
`paper-figures` also compares the rebuilt outputs against the paper's reference
CSVs and tables. Neither command reruns the simulation.

The `benchmarks`, `dimension-sweep`, `size-sweep`, and `sonar` commands hold exact
sampling artifacts fixed and regenerate private randomness and assignments.
By default the Sonar command also holds the original
primary Sonar null sample fixed. `--regenerate-null` reconstructs that sample
from its original seed, rather than choosing a new sample.

An experiment using new seeds or new sampled records is a new statistical
replication, not an exact replay of the realized paper experiment. It should
not be expected to reproduce every printed decimal.
