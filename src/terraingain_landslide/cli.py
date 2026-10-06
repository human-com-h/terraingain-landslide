"""Inactive scientific entry points and read-only scalar displays."""
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path
from .configuration import REGIONS, paths, protocol

def parser():
    p = argparse.ArgumentParser(prog='terraingain', description='Paired regional comparison implementation and stored displays')
    sub = p.add_subparsers(dest='command')
    c = sub.add_parser('config', help='Print a fixed configuration')
    c.add_argument('--recipe', choices=['r0', 'r1'], required=True)
    r = sub.add_parser('results', help='List or show stored CSV rows')
    r.add_argument('--tables', type=Path, default=Path('results/tables'))
    r.add_argument('--table')
    d = sub.add_parser('display', help='Draw existing scalar values')
    d.add_argument('--tables', type=Path, required=True)
    d.add_argument('--output', type=Path, required=True)
    d.add_argument('--qa', type=Path)
    for name in ['train', 'evaluate', 'gate']:
        q = sub.add_parser(name, help='Inspect an inactive command; execution requires --execute')
        q.add_argument('--paths', type=Path, required=True)
        q.add_argument('--execute', action='store_true')
        if name == 'gate':
            q.add_argument('--output', type=Path, required=True)
            continue
        q.add_argument('--recipe', choices=['r0', 'r1'], required=True)
        if name == 'train':
            q.add_argument('--shard', choices=REGIONS, required=True)
            q.add_argument('--seed', type=int, choices=[17, 29, 43], required=True)
            q.add_argument('--allocation-seconds', type=float)
        else:
            q.add_argument('--mode', choices=['audit', 'infer'], default='audit')
            q.add_argument('--session-hours', type=float, default=6)
    return p

def main(argv=None):
    started = time.monotonic()
    p = parser()
    a = p.parse_args(argv)
    if a.command is None:
        p.print_help()
        return 0
    if a.command == 'config':
        print(json.dumps(protocol(a.recipe), indent=2))
        return 0
    if a.command == 'results':
        if a.table:
            if Path(a.table).name != a.table or not a.table.endswith('.csv'):
                p.error('Use a CSV filename from the table directory')
            with (a.tables / a.table).open(encoding='utf-8-sig', newline='') as f:
                print(f.read())
        else:
            for file in sorted(a.tables.glob('*.csv')):
                with file.open(encoding='utf-8-sig', newline='') as f:
                    count = len(list(csv.DictReader(f)))
                print(file.name, count, 'rows')
        return 0
    if a.command == 'display':
        from .display import render
        render(a.tables, a.output, a.qa)
        return 0
    values = paths(a.paths)
    recipe = 'r1' if a.command == 'gate' else a.recipe
    pr = protocol(recipe)
    if not a.execute:
        print(json.dumps({'command':a.command, 'recipe':recipe, 'execute':False,
                          'paths':values}, indent=2))
        return 0
    from .utils.integrity import verify_source_tree
    verify_source_tree()
    env = dict(os.environ, TERRAINGAIN_ENCODER=values['encoder'],
               S05G_R0_REFERENCE=values['r0_reference'])
    env['S05F_DATA' if recipe == 'r0' else 'S05G_DATA'] = values[
        'training_data' if a.command == 'train' else 'outer_data']
    previous = [x for value in values['previous_outputs'] for x in ['--input', value]]
    if a.command == 'gate':
        from .runtime.r1.launcher.s05g_controls import build_stage17_gate
        a.output.parent.mkdir(parents=True, exist_ok=True)
        build_stage17_gate(values['previous_outputs'], a.output)
        return 0
    module = 'terraingain_landslide.runtime.' + recipe + '.launcher.'
    if a.command == 'evaluate':
        command = [sys.executable, '-m', module+'outer_runner', a.mode,
                   '--work',values['work'], '--export-dir',values['export_dir'],
                   '--session-hours',str(a.session_hours), *previous]
        return subprocess.call(command, env=env)
    hours = 6 if recipe == 'r0' else pr['execution']['caps']['pair_dual_T4_session_hours']
    command = [sys.executable, '-m', module+'shard_runner', 'train',
               '--shard',a.shard, '--seed',str(a.seed), '--work',values['work'],
               '--export-dir',values['export_dir'], '--input',values['training_data'],
               *previous, '--authorize-training']
    if recipe == 'r0':
        command += ['--session-hours',str(hours)]
        return subprocess.call(command,env=env)
    cap = a.allocation_seconds if a.allocation_seconds is not None else hours*3600
    # Check only the assigned pair when deciding whether an allocation is a resume.
    for root in values['previous_outputs']:
        if any(Path(root).rglob('S05G_'+a.shard+'_seed'+str(a.seed)+'_resume.zip')) and a.allocation_seconds is None:
            p.error('A resume requires --allocation-seconds')
        for file in Path(root).rglob('INDEX.json'):
            index = json.loads(file.read_text(encoding='utf-8-sig'))
            if index.get('shard') == a.shard and index.get('execution_seed') == a.seed and a.allocation_seconds is None:
                p.error('A resume requires --allocation-seconds')
    remaining=cap-(time.monotonic()-started)
    if remaining<=1800 or cap>hours*3600:
        p.error('Allocation must fit the original cap and leave the export reserve')
    command += ['--session-hours',str(remaining/3600)]
    from .runtime.r1.launcher.budget_guard import guarded_run
    os.environ.update(env)
    guarded_run(command,cap,started,'S05G-'+a.shard+'-seed'+str(a.seed),
                values['export_dir'])
    return 0

