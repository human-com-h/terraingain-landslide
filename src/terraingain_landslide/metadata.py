"""Checks for synthetic or user-supplied recovery metadata; never load tensors."""
from pathlib import PurePosixPath

def recovery_pointer(pointer, expected_config, file_sha256):
    name = PurePosixPath(pointer['file'])
    if (name.is_absolute() or len(name.parts) != 1 or '..' in name.parts
            or '\\' in pointer['file'] or ':' in pointer['file']
            or not name.name.startswith('resume_') or name.suffix != '.pt'):
        raise ValueError('Unsafe recovery filename')
    if (pointer['config_sha256'] != expected_config
            or pointer['sha256'] != file_sha256
            or type(pointer['step']) is not int or not 0 <= pointer['step'] <= 20000):
        raise ValueError('Recovery metadata identity mismatch')
    return True

