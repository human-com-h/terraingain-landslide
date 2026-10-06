"""Load portable paths and fixed runtime settings without model imports."""
from pathlib import Path
from .utils.io import read

REGIONS = ('dominicamaria', 'italy', 'hiroshima', 'hokkaido', 'thrissur')

def protocol(recipe):
    if recipe not in ('r0', 'r1'):
        raise ValueError('Unknown recipe')
    name = 'S05F_PROTOCOL.json' if recipe == 'r0' else 'S05G_PROTOCOL.json'
    file = Path(__file__).parent / 'runtime' / recipe / 'core' / name
    p = read(file)
    expected = 6.25e-6 if recipe == 'r0' else 3.125e-5
    if p['recipe']['optimization']['encoder_lr'] != expected:
        raise ValueError('Encoder rate differs from the fixed recipe')
    if (p['regions'] != list(REGIONS) or p['seeds'] != [17, 29, 43]
            or p['attempts'] != 20000 or p['recipe']['threshold'] != 0.5):
        raise ValueError('Protocol identity differs')
    return p

def paths(file):
    file = Path(file).resolve()
    p = read(file)
    required = {'training_data', 'outer_data', 'encoder', 'previous_outputs',
                'work', 'export_dir', 'r0_reference'}
    if set(p) != required or not isinstance(p['previous_outputs'], list):
        raise ValueError('Path configuration keys differ from the example')
    def absolute(value):
        value = Path(value)
        return str((file.parent / value).resolve() if not value.is_absolute() else value.resolve())
    return {k: [absolute(x) for x in v] if k == 'previous_outputs' else absolute(v)
            for k, v in p.items()}

