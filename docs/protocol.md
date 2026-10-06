# Protocol

The downstream model is SatMAE multispectral ViT-Base/8 with the original segmentation decoder. Inputs have shape 10×128×128 and band order B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12. The token groups are (0,1,2,6), (3,4,5,7), and (8,9). Positional embedding interpolation and scratch initialization remain as implemented in the frozen source.

Each procedure has five regional folds, seeds 17, 29, and 43, and two arms: 30 formal runs and 15 paired comparisons. This describes the formal matrix, not the complete history of platform executions or costs. The regions are dominicamaria, italy, hiroshima, hokkaido, and thrissur.

R0 uses encoder peak learning rate 6.25e−6; R1 uses 3.125e−5. Both use decoder peak learning rate 0.001, AdamW, weight decay 0.05, betas (0.9, 0.999), epsilon 1e−8, 3,333 warmup attempts, and final learning-rate ratio 0.01. The effective batch is eight, formed from four checkpointed microbatch graphs of size two. A single full-batch weighted BCE plus global Dice loss is differentiated; microbatch Dice losses are not averaged. The exact settings are in `configs/r0.yaml` and `configs/r1.yaml`.

Training starts from the official encoder subset for P and from the specified scratch initialization for S. The decoder is initialized identically within each pair. Optimizer, AMP, sampler, augmentation, and run state cannot be reused across seeds or arms. Positive/background sampling uses four of each with replacement. Sampler and augmentation generators use seed offsets 1000 and 3000; batch-wide right-angle rotation and horizontal flip follow the original exposure chain.

The primary endpoint is 20,000 attempted updates, including any AMP skips. Checkpoints are retained at 5k, 10k, 15k, and 20k. R1 stops if total skipped updates exceed 200. The fixed prediction threshold is sigmoid ≥ 0.5. Normalization uses fold training IDs only, float64 pixel moments, and a standard-deviation floor of 0.0001. The held region is physically absent from training packages.

Within each region and seed, positive-class F1 is calculated from pooled pixel TP/FP/FN/TN. Seeds are averaged within a region; the five regions then receive equal weight. Tile F1 averaging and pooling pixels across regions would change the endpoint.

The checkpoint choice compares the four stored source-validation checkpoints using exact rational regional F1 scores. Equal scores select the earliest checkpoint. The R0/R1 recipe choice compares only the fixed 20k candidates over the four source regions, using exact Fraction arithmetic, and selects R0 on a tie. A zero denominator contributes zero to the selection score while the raw metric remains null. P and S may choose different recipes; this endpoint compares strategies with 40,000 nominal candidate attempts per arm.

The outer barrier requires all 30 complete runs, 120 verified checkpoint states, committed source selections, matching run identities, and matched environments. Outer evaluation is separate from source fitting and selection. Historical outer caches had already informed development. Their reevaluation is not a new-event blind test.

The stored intervals use 5,120 m spatial blocks and 2,000 shared-weight replicates. They are conditional on the fixed caches, trained models, and three seeds. Preparation only retains their values. Unknown upstream geographic overlap and the historical development process limit interpretation. Same-ID role diagnostics also change training support, preprocessing fit, and models, so they do not identify a coverage mechanism. The results do not establish optimal model rankings or deployment readiness.

