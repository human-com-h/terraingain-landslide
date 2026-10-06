# TerrainGain Landslide

This repository contains the implementation and stored results of a paired comparison between pretrained (P) and scratch (S) encoders in a fixed landslide segmentation system. The gain is G = F1(P) − F1(S). Five historical regional caches and seeds 17, 29, and 43 were used. R0 and R1 differ in encoder peak learning rate; a separate procedure chooses between their fixed 20k candidates using source validation.

The package includes the SatMAE multispectral ViT-Base/8 adaptation, segmentation decoder, fold and preprocessing contracts, training and recovery code, identity checks, source selection, and separate outer evaluation. Stored tables retain all regions, seeds, checkpoint endpoints, and existing diagnostics. Images, labels, weights, checkpoints, and prediction arrays must be supplied separately.

## Install and inspect

Use Python 3.10 or later in a separate environment. The base package has no required GPU dependencies.

```text
python -m pip install --no-deps --no-build-isolation .
terraingain --help
terraingain config --recipe r0
terraingain results --tables results/tables
python -m unittest discover -s tests -v
```

The metadata tests need NumPy. Plotting needs Matplotlib. Optional dependencies can be installed with `pip install ".[engineering,display]"` if they are not already present. The historical training environment is specified separately in [docs/environment.md](docs/environment.md).

## Display the stored results

```text
terraingain results --tables results/tables --table Table1_primary_endpoints.csv
terraingain display --tables results/tables --output work/displays
```

The display command reads existing tables and draws their stored values. It does not recalculate F1, averages, intervals, or selections. The included figures are in [results/figures](results/figures); definitions and interpretation limits are in [docs/results.md](docs/results.md).

## Reproduce the original workflow

Copy `configs/paths.example.yaml` to a local path configuration and provide the inputs described in [docs/data.md](docs/data.md). Then inspect a pair command:

```text
terraingain train --recipe r0 --paths configs/paths.example.yaml --shard dominicamaria --seed 17
```

Execution requires an explicit `--execute` flag, the frozen Linux dual-T4 environment, verified fold data, and the recorded encoder subset. See [docs/reproduction.md](docs/reproduction.md) for the source gate, recovery, and outer barrier. The notebooks provide the same disabled-by-default entry points.

## Validation and status

This local candidate was checked for syntax, package structure, configuration loading, help pages, synthetic metadata contracts, and result-copy consistency. The stored displays were rendered and inspected. No training, neural-network forward pass, inference, prediction recount, resampling, or state deserialization was performed during preparation. These checks do not establish runtime or experimental reproduction.

The implementation was relocated into a package, with path and import changes. It has a new software identity. Historical output hashes refer to the original implementation; this package must not silently resume those outputs. The available returns bind results to the declared frozen core, but do not independently authenticate every executed platform source version.

The authors' original code and documentation use the [MIT License](LICENSE). SatMAE-derived components retain CC BY-NC 4.0, including its noncommercial condition. See [LICENSE_SCOPE.md](LICENSE_SCOPE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for scope and third-party terms.

This is a local candidate. Permissions for derived inputs and stored results, and a remote repository address, remain to be settled. See [AUTHORS.md](AUTHORS.md) for attribution.

