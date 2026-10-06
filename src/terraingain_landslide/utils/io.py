from pathlib import Path
import json, hashlib, os

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p, d):
    p = Path(p)
    p.parent.mkdir(exist_ok=True, parents=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(d, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    q.replace(p)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

