# Release validation

Validation date: November 20, 2026.

The public package was installed into a fresh CPython 3.12.13 environment on
Linux x86_64. This exercised compatible versions newer than parts of the original
production environment: NumPy 2.5.5, SciPy 1.18.1, pandas 3.1.1, matplotlib
3.11.2, and pytest 9.2.0. All numerical checks ran on a compute node.

The following checks passed:

- 49 automated tests, covering all four mechanisms and both objectives,
  frozen-implementation calibration parity, radial noise magnitude, distance
  preservation, degenerate public geometry, invalid inputs, and endpoint reuse.
- Sequential and two-worker Iris runs produced identical summary values and
  every stored assignment/array. Resuming preserved results; changed scientific
  settings were rejected instead of mixed into the same directory.
- Paper workload plans matched the original task counts: 1,600 instances per
  objective for all benchmarks, 1,000 Banknote instances, 28,000 Fashion instances
  plus 100 projection bases, and 1,000 Sonar repetition tasks.
- End-to-end quick runs completed for both Iris objectives, the Banknote size
  sweep, Fashion-MNIST dimension sweep, and Sonar rank experiment.
- The saved-results exporter produced 8,476 tidy summary rows. CSV-only plotting
  rendered all five experiment/objective groups.
- Rebuilding from saved results reproduced the eight original paper figures,
  their underlying numeric summaries, all printed appendix table rows, and the
  reported numerical quantities, with the documented Ecoli prose rounding
  discrepancy retained in the comparison report.
- All sixteen raw datasets and 360 sampled input masters passed source-row,
  partition, normalization, and reconstruction checks. All 611 immutable
  scientific assets matched the bundled SHA256 manifest.
- A source distribution and wheel built successfully. The wheel was installed
  into a separate directory and imported outside the checkout. Its bundled
  assets verified, fresh Banknote and Sonar downloads matched their checksums,
  the quick Sonar experiment completed, and saved-result export and CSV plotting
  worked without importing code from the original project. The wheel contains
  no Python bytecode caches.

After adding the projected-report objective guard and run-directory selection
checks, all 43 affected mechanism and sequential/parallel workflow tests passed
again. The remaining workflows were unchanged from the complete 49-test pass.

Quick runs validate execution and integration, not the full paper's statistical
claims. The complete production simulations were not repeated for this release.
Their original saved outcomes, configurations, seed namespaces, and unchanged
scientific code are included. Cross-platform QR/SVD and assignment ties can
change floating-point details; the recorded production summaries are the
reference for the paper's numerical claims.

The repository's CI runs the tests on Linux, macOS, and Windows with Python 3.11
and 3.12. Consult the current GitHub Actions status for those runs; the local
release checks described above were performed on Linux.
