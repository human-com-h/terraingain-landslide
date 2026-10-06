# Third-party components

The SatMAE dense encoder adaptation and positional embedding utilities derive from [SatMAE](https://github.com/sustainlab-group/SatMAE), commit `0b210aceb37a14bbbd897110db5b104b3271d818`. Attribution: Yezhen Cong, Samar Khanna, Chenlin Meng, Patrick Liu, Erik Rozi, Yutong He, Marshall Burke, David B. Lobell, and Stefano Ermon. The original code uses CC BY-NC 4.0; its unmodified license text is in `licenses/SatMAE.txt`.

`src/terraingain_landslide/models/model_g1v2.py` preserves the study's dense adaptation: grouped optical tokens, a fixed segmentation decoder, interpolation to the 16×16 patch grid, and the existing PyTorch attention path. `official_pos_embed.py` preserves the positional utilities and their upstream source links. These are adaptations rather than an official SatMAE segmentation product.

The [weight record](https://zenodo.org/records/7338613) identifies CC BY 4.0 for the checkpoint. The code and weight licenses differ. No checkpoint is included here.

PyTorch, torchvision, NumPy, Matplotlib, and their dependencies are external software. Their own licenses apply. They are not vendored into this repository.

The authors' original code and documentation use the MIT License. The SatMAE-derived files listed above remain subject to CC BY-NC 4.0; MIT does not override their upstream terms. Stored result tables and figures have separate, unresolved sharing permissions. See [LICENSE_SCOPE.md](LICENSE_SCOPE.md) for the boundaries of the MIT grant.

