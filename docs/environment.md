# Environments

The accepted returns record Linux on Kaggle, two Tesla T4 GPUs per pair, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, NumPy 2.0.2, CUDA 12.8, and cuDNN 91002. Pair and resume checks compare torch, torchvision, NumPy, CUDA, cuDNN, and GPU identities exactly. R1 also compares against the recorded R0 environment. Internet access was disabled. A local installation with newer packages is not the recorded training environment.

The launcher requires at least 12 GiB free on each T4. Its disk check estimates model and optimizer states, eight retained checkpoint states per pair, and export copies, with metadata and margin. Use the calculated requirement, rather than assuming that free space for one model is enough.

The earlier local engineering record used Python 3.12.3, torch 2.2.2, torchvision 0.17.2, NumPy 1.26.4, CUDA 11.8, and an RTX 4060 Laptop GPU. It was a separate check environment. Old timm 0.3.2 was used for an implementation comparison and is not a training dependency.

The present preparation used Python 3.12.9 on Windows, NumPy 2.4.6, and Matplotlib 3.10.0 for syntax, package checks, CPU metadata fixtures, and displays. PyMuPDF 1.28.2 was used from an external check directory to inspect PDFs. CUDA, model forwards, and state loading were not validated. Command results are retained in the external preparation record. GPU dependencies are not installed by the base package or its tests.

