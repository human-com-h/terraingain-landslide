# Inputs

The parent dataset is [Sen12Landslides](https://huggingface.co/datasets/paulhoehn/Sen12Landslides). The study used five fixed 256-tile caches on the 10 m grid. The cache selection used coordinate-based spatial strata and fixed hash ordering; invalid dates and bounds were handled during the original preparation. The parent release does not by itself guarantee byte-identical reconstruction of every study cache. Obtain the frozen caches, partitions, and package manifests from the authors.

`configs/folds/` preserves the ID lists and their order. `configs/fits/` preserves train-only moments, class weights, and positive/background pools. These records contain no images or labels. `configs/FOLD_SUMMARY.json` and `SOURCE_CENSUS.json` record their support sizes. Runtime `INPUT_MANIFEST.json` files retain array and metadata hashes. Complete private source mappings are kept outside this repository.

A training directory contains `DATA_MANIFEST.json`, `FOLD.json`, `FIT.json`, and `source_optical/<region>.npz` plus `<region>.json` for the four source regions. The NPZ contains x with shape (256,10,128,128) and binary y with shape (256,128,128); JSON records name, region, positive pixels, and coordinates. The runtime checks the exact file inventory, bytes, ID order, and train-only fit. The held region must be absent from every mounted training input, including archives.

The separate outer directory contains the five cache pairs, `folds/`, `fits/`, and its own `DATA_MANIFEST.json`. It must never be used as a training input. Expected training and outer manifest hashes are in runtime `SOURCE_PACKS.json` files. No array is included in this repository.

The official [SatMAE source](https://github.com/sustainlab-group/SatMAE) is recorded at commit `0b210aceb37a14bbbd897110db5b104b3271d818`. The [multispectral checkpoint](https://zenodo.org/records/7338613) is `pretrain-vit-base-e199.pth`, 1,338,694,433 bytes, MD5 `1269e6a8255cee0affdeb6bb75776c86`, SHA256 `cfce145a0c88ee7f85ab639158d2456319b264d1560185af4f8f021c79d4ed19`.

The required encoder-only subset is `satmae_base8_encoder.pt`, SHA256 `a93bc2884d5a0896285a5b7d6e7ca726c39d532be87dc2c6725d1c0797abee1f`. Its 155 keys and tensor shapes are recorded in `configs/weight_provenance.json`. It is an unchanged tensor subset of the official checkpoint. Position interpolation happens during loading. Obtain that verified subset separately; serialization in another environment is not assumed to recreate the same file hash. No weight extraction or download runs during installation or testing.

The source-only R0 20k reference in `results/source_reference/` contains stored confusion counts and no outer scores. It supports the retained source selection code. Raw predictions and full training states remain external inputs.

The parent data terms and permissions for derived caches, labels, weights, and result redistribution must be checked before distribution. This candidate does not assign a dataset license or create a download service.

