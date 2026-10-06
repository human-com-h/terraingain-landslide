"""Separate outer-only instance: global audit barrier, inference, independent final assessment."""
import sys,time,os,zipfile,shutil,argparse,gc
from .transport import *
sys.path.insert(0,str(CORE))
REGIONS=('dominicamaria','italy','hiroshima','hokkaido','thrissur')

def assemble(inputs,work):
 root=Path(work)/'results';root.mkdir(parents=True,exist_ok=True);mapping={};heads={}
 for r in REGIONS:
  sources=[source for seed in (None,*SEEDS) if (source:=latest(inputs,r,seed)) is not None]
  require(bool(sources),'Missing completed shard '+r)
  allfiles={};owners={};records=[]
  for p,idx,h in sources:
   seed=idx.get('execution_seed');allowed=set(runs(r,seed))
   require(set(idx['complete']).issubset(allowed),'Unexpected run receipt')
   if seed is not None:require(set(idx['complete'])==allowed,'Independent seed pair incomplete')
   for run_id in idx['complete']:
    require(run_id not in owners,'Duplicate run ownership: '+run_id)
    owners[run_id]=h
   records.append(dict(execution_seed=seed,index_sha256=h,index_path=str(p),generation=idx['generation']))
   with index_archive(p) as z:
    for rel,record in idx['files'].items():
     require(rel not in allfiles,'Overlapping execution metadata: '+rel);allfiles[rel]=record;out=safe(root,rel)
     if record['kind']=='metadata':
      if out.exists():require(sha(out)==record['sha256'],'Existing merge metadata changed')
      else:
       out.parent.mkdir(parents=True,exist_ok=True)
       with z.open('results/'+rel) as src,out.open('wb') as dst:shutil.copyfileobj(src,dst)
     else:
      require('/checkpoint_' in rel and rel.endswith('/state.pt'),'Completed export contains rolling/orphan state')
      mapping[str(out.resolve())]=resolve_blob(record,inputs)
  require(set(owners)==set(runs(r)),'All 30 runs must complete before any outer inference: '+r)
  if len(records)==1 and records[0]['execution_seed'] is None:heads[r]=records[0]
  else:
   key=[dict(execution_seed=v['execution_seed'],index_sha256=v['index_sha256']) for v in records]
   composite=__import__('hashlib').sha256(__import__('json').dumps(key,sort_keys=True,separators=(',',':')).encode()).hexdigest()
   heads[r]=dict(index_sha256=composite,sources=records)
  require(set(complete_runs(root,r,allfiles))==set(runs(r)),'Receipt inventory mismatch')
 require(len(mapping)==120 and len(list(root.rglob('result.json')))==30,'Exactly 30 runs / 120 states required')
 return root,mapping,heads

def restore_outer(inputs,work):
 target=Path(work);target.mkdir(parents=True,exist_ok=True)
 # Outer exports are monotonically growing immutable files. Union is allowed only with identical overlaps.
 for source in inputs:
  for p in Path(source).rglob('S05G_outer_resume.zip'):
   with zipfile.ZipFile(p) as z:
    require(len(z.namelist())==len(set(z.namelist())),'Duplicate outer archive members')
    index=__import__('json').loads(z.read('EXPORT_MANIFEST.json'))
    require(set(z.namelist())==set(index)|{'EXPORT_MANIFEST.json'},'Outer export inventory mismatch')
    for rel,h in index.items():
     require(rel.startswith(('outer/','results/','AUDIT_FAILURE_')) or rel in ('TRAINING_REGISTRY.json','FULL_STATE_VERIFICATION.json','FINAL_ASSESSMENT.json','INPUT_ORIGINS.json','SOURCE_RECIPE_SELECTION.json'),'Unexpected outer export member')
     out=safe(target,rel)
     with z.open(rel) as f:require(digest(f)==h,'Corrupt outer resume')
     if out.exists():require(sha(out)==h,'Forked outer result')
     else:
      out.parent.mkdir(parents=True,exist_ok=True)
      with z.open(rel) as src,out.open('wb') as dst:shutil.copyfileobj(src,dst)

def export_outer(work,dest):
 work=Path(work);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);target=dest/'S05G_outer_resume.zip';tmp=target.with_suffix('.zip.tmp')
 files={p.relative_to(work).as_posix():sha(p) for p in work.rglob('*') if p.is_file() and p.suffix!='.pt' and not any(v.endswith('.tmp') for v in p.relative_to(work).parts) and p.name!='RUNNING.lock'}
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
  for rel in files:z.write(work/rel,rel)
  z.writestr('EXPORT_MANIFEST.json',__import__('json').dumps(files,sort_keys=True))
 with zipfile.ZipFile(tmp) as z:
  for rel,h in files.items():
   with z.open(rel) as f:require(digest(f)==h,'Export roundtrip mismatch')
 tmp.replace(target);write(dest/'S05G_outer_SHA256.json',{target.name:sha(target)});return target

def run(inputs,work,dest,hours,allow_inference):
 from ..core.common_g1v2 import verify_data,environment
 from ..core.controlled import env_identity,model_for
 from ..core.assess_s05f import audit_training
 from .shard_runner import lock
 import torch
 work=Path(work);work.mkdir(parents=True,exist_ok=True)
 with lock(work):
  try:
   verify_delivery();data=verify_data(outer=True);restore_outer(inputs,work);deadline=time.time()+hours*3600-1800
   root,mapping,heads=assemble(inputs,work);registry_path=work/'TRAINING_REGISTRY.json'
   if registry_path.exists():
    registry=read(registry_path);require(registry['input_heads']=={r:v['index_sha256'] for r,v in heads.items()},'Different training histories after outer unlock')
    evidence=read(work/'FULL_STATE_VERIFICATION.json');require(evidence['status']=='PASS' and evidence['count']==30 and evidence['state_blobs']==120 and evidence['registry_sha256']==sha(registry_path),'Invalid global audit evidence')
   else:
    with virtual_states(mapping):registry=audit_training(root,data)
    registry['input_heads']={r:v['index_sha256'] for r,v in heads.items()};write(registry_path,registry)
    write(work/'INPUT_ORIGINS.json',dict(shards=heads,blobs={Path(k).relative_to(root.resolve()).as_posix():str(v) for k,v in mapping.items()},note='Retain these actual Kaggle Dataset/Output versions; paths are provenance, content hashes define identity.'))
    write(work/'FULL_STATE_VERIFICATION.json',dict(status='PASS',count=30,state_blobs=120,selections=30,registry_sha256=sha(registry_path),identity=identity(),automatic_followon=False))
   from .bounded_selection import seal_selection
   seal_selection(root,registry,work/'SOURCE_RECIPE_SELECTION.json')
   require(registry['status']=='PASS' and registry['count']==30 and registry['state_blobs']==120 and registry['selections']==30,'Global barrier not passed')
   if not allow_inference:print('GLOBAL AUDIT PASS; inference not requested');return
   from .s05g_controls import require_parent_environment
   require_parent_environment()
   require(torch.cuda.is_available(),'Outer inference requires CUDA');require(env_identity(environment())==registry['environment'],'Inference runtime differs from frozen training runtime')
   registry['_file_sha256']=sha(registry_path);outer=work/'outer'
   from ..core.train_s05f import evaluate
   import numpy as np
   for r in REGIONS:
    records=read(data/'source_optical'/f'{r}.json')['records'];names=[m['name'] for m in records];fit=read(data/'fits'/f'{r}.json')
    require(names==read(data/'folds'/f'{r}.json')['outer_test_ids'],'Outer ID order changed')
    with np.load(data/'source_optical'/f'{r}.npz',allow_pickle=False) as z:x=z['x'];y=z['y']
    for s in SEEDS:
     for a in ARMS:
      run_id=f'{r}/seed{s}/{a}'
      for t in (5000,10000,15000,20000):
       cp=outer/run_id/f'checkpoint_{t:05d}';statehash=registry['runs'][run_id]['states'][str(t)]
       if cp.exists():
        m=read(cp/'MANIFEST.json');require(m==dict(state_sha256=statehash,registry_sha256=registry['_file_sha256'],prediction_sha256=sha(cp/'outer.npz')),'Changed committed outer prediction');continue
       if time.time()>=deadline:print('Outer session paused safely');return
       model=model_for(a,s).cuda()
       with virtual_states(mapping):state=torch.load(root/run_id/f'checkpoint_{t:05d}'/'state.pt',map_location='cpu',weights_only=False)
       require(state['step']==t,'State step mismatch');model.load_state_dict(state['model']);del state;model.eval()
       stage=cp.with_name(cp.name+'.tmp');stage.mkdir(parents=True,exist_ok=True)
       evaluate(model,x,y,np.arange(len(names)),np.array([r]*len(names)),names,fit,stage/'outer.npz')
       write(stage/'MANIFEST.json',dict(state_sha256=statehash,registry_sha256=registry['_file_sha256'],prediction_sha256=sha(stage/'outer.npz')));stage.replace(cp)
       del model;gc.collect();torch.cuda.empty_cache();print('OUTER COMMITTED',run_id,t,flush=True)
   result=dict(status='EXTENSION_PREDICTIONS_COMPLETE',experiment='S05G',decision=None,registry_sha256=sha(registry_path),source_recipe_selection_sha256=sha(work/'SOURCE_RECIPE_SELECTION.json'),outer_predictions=120,original_S05F_decision_unchanged='F4');write(work/'FINAL_ASSESSMENT.json',result);print('FINAL',result['status'])
  except BaseException as exc:
   import traceback
   message=str(exc);status='INCOMPLETE' if any(v in message for v in ('Missing completed shard','All 30 runs','Exactly 30','Need exactly 120','Missing historical output')) else 'TECHNICAL_FAILURE'
   write(work/f'AUDIT_FAILURE_{time.time_ns()}.json',dict(status=status,decision=None,error=repr(exc),traceback=traceback.format_exc(),automatic_followon=False))
   raise
  finally:export_outer(work,dest)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','infer','export']);p.add_argument('--input',type=Path,action='append',default=[]);p.add_argument('--work',type=Path,required=True);p.add_argument('--export-dir',type=Path,required=True);p.add_argument('--session-hours',type=float,default=6);a=p.parse_args()
 require(.5<a.session_hours<=6,'Session budget must be (.5,6] hours')
 if a.mode=='export':export_outer(a.work,a.export_dir)
 else:run(a.input,a.work,a.export_dir,a.session_hours,a.mode=='infer')
if __name__=='__main__':main()
