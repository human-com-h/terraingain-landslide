"""S05G identity/source gate checks. No training on import."""
import json,hashlib,os
from pathlib import Path

CORE=Path(__file__).resolve().parents[1]/'core'
REGIONS=('dominicamaria','italy','hiroshima','hokkaido','thrissur')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require_parent_environment():
 from ..core.common_g1v2 import environment
 from ..core.controlled import env_identity
 expected=read(CORE/'S05G_PROTOCOL.json')['parent']['R0_environment_identity']
 actual=env_identity(environment())
 if actual!=expected:raise RuntimeError('R1 runtime identity differs from original R0. Stop; do not install/change packages to continue.')
 return actual

def validate_stage17_gate(inputs):
 expected_protocol=sha(CORE/'S05G_PROTOCOL.json');candidates=[]
 for root in inputs:
  candidates.extend(Path(root).rglob('S05G_STAGE17_GATE.json'))
 if not candidates:raise RuntimeError('Stage2 requires the completed five-region Stage17 CPU gate. No training launched.')
 unique={sha(p):p for p in candidates}
 if len(unique)!=1:raise RuntimeError('Conflicting Stage17 gates')
 gate=read(next(iter(unique.values())))
 if gate.get('status')!='SOURCE_ENGINEERING_GATE_VERIFIED' or gate.get('protocol_sha256')!=expected_protocol:
  raise RuntimeError('Invalid Stage17 gate identity')
 if gate.get('core_manifest_sha256')!=sha(CORE/'PACKAGE_MANIFEST.json'):
  raise RuntimeError('Stage17 gate belongs to a different runtime')
 expected={f'{r}/seed17/{a}' for r in REGIONS for a in ('pretrained','scratch')}
 if set(gate.get('runs',{}))!=expected or gate.get('outer_read') is not False:
  raise RuntimeError('Stage17 gate inventory/scope mismatch')
 for item in gate['runs'].values():
  if item['successful_updates']+item['skipped_updates']!=20000 or item['skipped_updates']>200:
   raise RuntimeError('Stage17 source numerical gate failed')
 return gate

def build_stage17_gate(inputs,output):
 from .transport import latest,index_archive,digest,read as transport_read
 protocol=read(CORE/'S05G_PROTOCOL.json');records={};heads={}
 for held in REGIONS:
  head=latest(inputs,held,17)
  if head is None:raise RuntimeError('Missing Stage17 pair: '+held)
  path,index,head_hash=head;expected={f'{held}/seed17/{a}' for a in ('pretrained','scratch')}
  if set(index['complete'])!=expected:raise RuntimeError('Incomplete Stage17 pair: '+held)
  if any(Path(rel).name.startswith('failure_') for rel in index['files']):raise RuntimeError('Preserved technical failure: '+held)
  heads[held]=dict(index_sha256=head_hash,execution_seed=17)
  with index_archive(path) as z:
   for run in sorted(expected):
    def item(name):return json.loads(z.read('results/'+run+'/'+name))
    config=item('config.json');receipt=item('result.json')
    if (config['split'],config['seed'],config['arm'])!=(held,17,run.split('/')[-1]):
     raise RuntimeError('Stage17 run name/config mismatch')
    if config['protocol_sha256']!=sha(CORE/'S05G_PROTOCOL.json') or config['package_sha256']!=sha(CORE/'PACKAGE_MANIFEST.json'):
     raise RuntimeError('Stage17 run identity mismatch')
    if config['environment_identity']!=protocol['parent']['R0_environment_identity']:
     raise RuntimeError('Stage17 environment mismatch')
    bindings=read(CORE/'S05G_FOLD_BINDINGS.json')[held]
    if config['split_sha256']!=bindings['fold_sha256'] or config['fit_sha256']!=bindings['fit_sha256'] or config['data_manifest_sha256']!=bindings['data_manifest_sha256']:
     raise RuntimeError('Stage17 frozen fold/fit/data binding mismatch')
    if receipt['status']!='COMPLETE' or receipt['primary_step']!=20000 or receipt['checkpoint_set']!=[5000,10000,15000,20000]:
     raise RuntimeError('Stage17 receipt incomplete')
    required={f'checkpoint_{step:05d}/{name}' for step in (5000,10000,15000,20000) for name in ('state.pt','train.npz','validation.npz','checkpoint.json','MANIFEST.json')}
    required|={'SELECTION.json','INITIALIZATION.json','config.json','curve.json'}
    if set(receipt['files'])!=required:raise RuntimeError('Stage17 checkpoint evidence incomplete')
    if receipt['successful_updates']+receipt['skipped_updates']!=20000 or receipt['skipped_updates']>200:
     raise RuntimeError('Stage17 AMP numerical gate failed')
    for rel,h in receipt['files'].items():
     index_record=index['files'][run+'/'+rel]
     if index_record['sha256']!=h:raise RuntimeError('Receipt/index hash mismatch')
     if rel.endswith('/state.pt') and index_record['kind']!='blob':raise RuntimeError('Checkpoint must reference a full state blob')
    records[run]=dict(config_sha256=index['files'][run+'/config.json']['sha256'],
       receipt_sha256=index['files'][run+'/result.json']['sha256'],
       successful_updates=receipt['successful_updates'],skipped_updates=receipt['skipped_updates'])
 gate=dict(status='SOURCE_ENGINEERING_GATE_VERIFIED',protocol_sha256=sha(CORE/'S05G_PROTOCOL.json'),
  core_manifest_sha256=sha(CORE/'PACKAGE_MANIFEST.json'),runs=records,input_heads=heads,
  source_engineering_only=True,outer_read=False,state_payloads_loaded=0,
  not_full_state_verification=True,not_f1_or_gain_gating=True)
 Path(output).write_text(json.dumps(gate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 return gate
