# Running the workflow

The default CLI and notebooks are inactive. Help pages and command inspection require no data, weights, or GPU. Training and outer evaluation require an explicit `--execute` flag. Use the exact environment and verified inputs described in `environment.md` and `data.md`.

Paths in a local configuration are resolved relative to that configuration. Use absolute paths when inputs reside elsewhere. `configs/paths.example.yaml` lists the required keys. The example uses JSON syntax, which is valid YAML; keep that syntax for the lightweight configuration loader.

```text
terraingain config --recipe r0
terraingain train --recipe r0 --paths paths.local.yaml --shard dominicamaria --seed 17
terraingain train --recipe r1 --paths paths.local.yaml --shard dominicamaria --seed 17
```

Appending `--execute` starts the assigned pair only. Each region/seed requires a distinct work and export directory. GPU0 handles P and GPU1 handles S. Do not assign the same pair to multiple sessions. Preserve all output, including failure records. There is no automatic retry or next-seed launch.

R0 retains the independent-seed transport corrections and its six-hour session limit. R1 begins with all five seed17 pairs. Before seed29 or seed43, its source metadata gate must confirm the five completed seed17 pairs, frozen fold/fit/data bindings, matching runtime identity, 20k completion, and AMP skip limits:

```text
terraingain gate --paths paths.local.yaml --output work/S05G_STAGE17_GATE.json
```

The gate command also requires `--execute` to read the supplied previous outputs. It does not use outer scores. A new R1 allocation has a 4.8-hour dual-T4 cap from CLI entry, including preparation, preflight, restore, and export. A resume requires a separate confirmed remaining allocation, specified with `--allocation-seconds`; no remaining budget is inferred from old records. The allocation guard and historical GPU preflight run only after explicit execution. They are not part of the lightweight test suite.

Resume input must belong to the same recipe, region, seed, configuration, environment, and runtime identity. Preserve the resume index, its origin sidecar when expanded, and all referenced state files. The rolling pointer is committed after the state and hash; a committed checkpoint can recover an interruption before the pointer update. Uncommitted temporary files are ignored. These rules are retained from the original implementation.

This package has new protocol and code manifest hashes because imports, paths, and administrative configuration fields changed. Historical resume outputs remain tied to the original frozen runtime and are not silently rebound. A fresh run under this candidate obtains its own identity. The original training core was selected using the freeze and accepted-return hashes; the returns do not separately authenticate every executed notebook source/version. GPU execution of the repackaged candidate is unverified.

After all runs and source selections are committed, outer evaluation uses a separate input configuration:

```text
terraingain evaluate --recipe r0 --paths paths.outer.yaml --mode audit
terraingain evaluate --recipe r1 --paths paths.outer.yaml --mode infer
```

These commands remain inactive until `--execute` is supplied. The historical audit mode deserializes full states and reads predictions; infer performs the barrier first and then outer inference. They were not run during preparation. Existing result displays should be reproduced with `terraingain display`, which only reads the stored scalar tables.

