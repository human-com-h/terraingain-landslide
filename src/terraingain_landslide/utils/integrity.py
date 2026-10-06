"""Code and external initializer checks. No deserialization."""
from pathlib import Path
from .io import read, sha

PACKAGE = Path(__file__).resolve().parents[1]
ENCODER_SHA256 = 'a93bc2884d5a0896285a5b7d6e7ca726c39d532be87dc2c6725d1c0797abee1f'

def verify_source_tree():
    for relative, expected in read(PACKAGE / 'CODE_MANIFEST.json')['files'].items():
        file = (PACKAGE / relative).resolve()
        if not file.is_relative_to(PACKAGE) or sha(file) != expected:
            raise RuntimeError('Code integrity mismatch: ' + relative)

def checked_encoder(path):
    path = Path(path)
    if not path.is_file() or sha(path) != ENCODER_SHA256:
        raise RuntimeError('Verified encoder subset required')
    return path

