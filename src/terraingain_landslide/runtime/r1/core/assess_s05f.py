"""Independent S05F evidence audit. Never trains or chooses new scientific settings."""
import hashlib,json,math
from pathlib import Path
from fractions import Fraction
import numpy as np
P=Path(__file__).resolve().parent
REG=['dominicamaria','italy','hiroshima','hokkaido','thrissur'];SEEDS=[17,29,43];ARMS=['pretrained','scratch'];STEPS=[5000,10000,15000,20000]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def met(c):
 tp,fp,fn,tn=map(int,c);ratio=lambda a,b:a/b if b else None
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,f1=ratio(2*tp,2*tp+fp+fn),iou=ratio(tp,tp+fp+fn),precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),fpr=ratio(fp,fp+tn),fnr=ratio(fn,tp+fn),predicted_positive_fraction=ratio(tp+fp,tp+fp+fn+tn),reference_positive_fraction=ratio(tp+fn,tp+fp+fn+tn))
def compare(a,b):
 if isinstance(a,dict):
  for k,v in a.items():compare(v,b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):compare(x,y)
 elif a is None:assert b is None
 elif isinstance(a,(float,np.floating)):assert isinstance(b,(int,float)) and math.isfinite(b) and abs(a-b)<1e-9,(a,b)
 else:assert a==b,(a,b)
def finite(v):
 if isinstance(v,dict):
  for x in v.values():finite(x)
 elif isinstance(v,(list,tuple)):
  for x in v:finite(x)
 elif isinstance(v,float):assert math.isfinite(v),'Unexpected NaN/Inf'
def aggregate(c,g):
 m=met(c.sum(0));m['regions']={r:met(c[g==r].sum(0)) for r in sorted(set(g))}
 for ep in ['f1','iou','precision','recall','fpr','fnr','predicted_positive_fraction']:
  a=[v[ep] for v in m['regions'].values()];m['inner_macro_'+ep]=float(np.mean(a)) if all(v is not None for v in a) else None
 return m

def source(data):
 manifest=read(P/'INPUT_MANIFEST.json');meta={};truth={}
 for r in REG:
  q=Path(data)/'source_optical'/f'{r}.npz';assert sha(q)==manifest['optical_arrays'][r]['sha256'];assert sha(q.with_suffix('.json'))==manifest['metadata'][r]['sha256']
  records=read(q.with_suffix('.json'))['records']
  with np.load(q,allow_pickle=False) as z:ys=z['y']
  assert ys.shape==(256,128,128) and np.isin(ys,[0,1]).all()
  for m,y in zip(records,ys):
   assert m['name'] not in meta and m['positive_pixels']==int(y.sum())
   meta[m['name']]=m;truth[m['name']]=y.astype(bool)
 return meta,truth

def recount(path,wanted,meta,truth):
 with np.load(path,allow_pickle=False) as z:
  assert set(z.files)=={'shape','prediction','truth','counts','names','groups'}
  names=z['names'].tolist();groups=z['groups'].astype(str);shape=tuple(z['shape']);assert names==wanted and shape==(len(wanted),128,128)
  assert groups.tolist()==[meta[n]['region'] for n in names]
  for key in ('prediction','truth'):assert z[key].dtype==np.uint8 and z[key].shape==(len(names),2048)
  a=np.unpackbits(z['prediction'],axis=1,count=16384).reshape(shape).astype(bool);b=np.unpackbits(z['truth'],axis=1,count=16384).reshape(shape).astype(bool)
  assert np.array_equal(b,np.stack([truth[n] for n in names]))
  c=np.column_stack([(a&b).sum((1,2)),(a&~b).sum((1,2)),(~a&b).sum((1,2)),(~a&~b).sum((1,2))]);assert np.array_equal(c,z['counts'])
 return c,groups

def selection(candidates,regions):
 scores={}
 for t in STEPS:
  parts=[]
  for r in regions:
   tp,fp,fn,tn=candidates[str(t)][r];assert all(type(v)==int and v>=0 for v in (tp,fp,fn,tn));d=2*tp+fp+fn;parts.append(Fraction(2*tp,d) if d else Fraction(0))
  scores[t]=sum(parts)/4
 best=max(scores.values());return min(t for t in STEPS if scores[t]==best),{str(t):dict(numerator=v.numerator,denominator=v.denominator) for t,v in scores.items()}

def audit_training(root,data):
 import torch,gc
 from terraingain_landslide.data.fold_contract import fit_train_only
 root=Path(root);data=Path(data);p=read(P/'S05G_PROTOCOL.json');identity=dict(protocol_sha256=sha(P/'S05G_PROTOCOL.json'),core_manifest_sha256=sha(P/'PACKAGE_MANIFEST.json'))
 for n,h in read(P/'PACKAGE_MANIFEST.json')['files'].items():assert sha(P/n)==h,n
 expected={f'{r}/seed{s}/{a}' for r in REG for s in SEEDS for a in ARMS}
 actual={f.parent.relative_to(root).as_posix() for f in root.rglob('result.json')};assert actual==expected,'INCOMPLETE: exactly 30 distinct runs required'
 assert not list(root.rglob('failure_*.json')),'Unresolved technical failure'
 meta,truth=source(data);fits={};folds={}
 for r in REG:
  fold=read(data/'folds'/f'{r}.json');fit=read(data/'fits'/f'{r}.json')
  def loader(region):
   assert region!=r
   with np.load(data/'source_optical'/f'{region}.npz',allow_pickle=False) as z:x=z['x'];y=z['y']
   return x,y,[m['name'] for m in read(data/'source_optical'/f'{region}.json')['records']]
  assert fit_train_only(fold,loader)==fit,'Fold fit mismatch';fits[r]=fit;folds[r]=fold
 chains={};rngs={}
 for r in REG:
  pos=np.array(fits[r]['positive_indices']);neg=np.array(fits[r]['negative_indices'])
  for s in SEEDS:
   sam=torch.Generator().manual_seed(s+1000);aug=torch.Generator().manual_seed(s+3000);h='00'*32
   for t in range(1,20001):
    ids=np.r_[pos[torch.randint(len(pos),(4,),generator=sam).numpy()],neg[torch.randint(len(neg),(4,),generator=sam).numpy()]].astype('<i8');rot=int(torch.randint(4,(1,),generator=aug));flip=int(torch.randint(2,(1,),generator=aug));h=hashlib.sha256(bytes.fromhex(h)+ids.tobytes()+bytes([rot,flip])).hexdigest()
    if t in STEPS:chains[r,s,t]=h;rngs[r,s,t]=(sam.get_state().clone(),aug.get_state().clone());assert h==read(P/'EXPECTED_EXPOSURE_CHAINS.json')[f'{r}/seed{s}'][str(t)]
 entries={};envs=[];inits={};signature=read(P/'MODEL_STRUCTURE.json')
 for run in sorted(expected):
  out=root/run;cfg=read(out/'config.json');res=read(out/'result.json');finite(cfg);finite(res);r=cfg['split'];s=cfg['seed'];a=cfg['arm'];assert run==f'{r}/seed{s}/{a}'
  assert cfg['protocol_sha256']==identity['protocol_sha256'] and cfg['package_sha256']==identity['core_manifest_sha256']
  assert cfg['input_manifest_sha256']==sha(P/'INPUT_MANIFEST.json') and cfg['split_sha256']==sha(data/'folds'/f'{r}.json') and cfg['fit_sha256']==sha(data/'fits'/f'{r}.json')
  assert cfg['data_manifest_sha256']==read(P/'SOURCE_PACKS.json')[r]
  e=cfg['environment_identity'];envs.append(e);assert e['torch']==p['recipe']['environment']['kaggle_torch'] and e['torchvision']==p['recipe']['environment']['kaggle_torchvision'] and e['gpu']=='Tesla T4'
  needed={'config.json','curve.json','INITIALIZATION.json','SELECTION.json'}|{f'checkpoint_{t:05d}/{n}' for t in STEPS for n in ('state.pt','train.npz','validation.npz','checkpoint.json','MANIFEST.json')}
  assert set(res['files'])==needed and res['status']=='COMPLETE' and res['primary_step']==20000 and res['checkpoint_set']==STEPS
  for n,h in res['files'].items():assert sha(out/n)==h,n
  curves=read(out/'curve.json');finite(curves);assert [v['attempt'] for v in curves]==list(range(500,20001,500))
  previous=0
  for v in curves:
   t=v['attempt'];assert v['successful_updates']+v['skipped_amp_updates']==t and previous<=v['skipped_amp_updates']<=t;previous=v['skipped_amp_updates'];assert v['loss_mean_last_interval']>=0 and v['cumulative_loss']>=0
  init=read(out/'INITIALIZATION.json');assert init['structure']==signature;inits[r,s,a]=init
  candidates={};states={};gaps={}
  for t in STEPS:
   cp=out/f'checkpoint_{t:05d}';m=read(cp/'MANIFEST.json');rec=read(cp/'checkpoint.json');finite(rec);assert m['step']==t and rec['step']==t and rec['primary']==(t==20000)
   assert m['config_sha256']==rec['config_sha256']==sha(out/'config.json');assert set(m['files'])=={'state.pt','train.npz','validation.npz','checkpoint.json'}
   for n,h in m['files'].items():assert h==res['files'][f'checkpoint_{t:05d}/{n}']
   assert rec['successful']+rec['skipped']==t and rec['skipped']==curves[t//500-1]['skipped_amp_updates'] and rec['exposure_sha256']==chains[r,s,t]
   state=torch.load(cp/'state.pt',map_location='cpu',weights_only=False)
   assert state['step']==t and state['config_sha256']==sha(out/'config.json') and state['skipped']==rec['skipped'] and state['exposure_sha256']==chains[r,s,t]
   assert {k:state['environment'][k] for k in e}==e
   assert {n:list(v.shape) for n,v in state['model'].items()}==signature
   assert all(torch.isfinite(v).all().item() for v in state['model'].values())
   assert set(state['rng'])=={'python','numpy','torch','cuda','sampler','augmentation'} and state['rng']['cuda']
   assert torch.equal(state['rng']['sampler'],rngs[r,s,t][0]) and torch.equal(state['rng']['augmentation'],rngs[r,s,t][1])
   assert state['optimizer']['state'] and len(state['optimizer']['param_groups'])==4 and state['scaler']['scale']>0
   recipe=p['recipe']['optimization'];warm=recipe['warmup_attempts'];factor=t/warm if t<=warm else recipe['final_lr_ratio']+(1-recipe['final_lr_ratio'])*(1+math.cos(math.pi*(t-warm)/(20000-warm)))/2
   for gi,group in enumerate(state['optimizer']['param_groups']):
    role='encoder' if gi<2 else 'decoder';assert group['role']==role;compare(group['initial_lr'],recipe[role+'_lr']);compare(group['lr'],recipe[role+'_lr']*factor);compare(group['weight_decay'],recipe['weight_decay'] if gi%2 else 0.);compare(list(group['betas']),recipe['betas']);compare(group['eps'],recipe['eps'])
   for st in state['optimizer']['state'].values():
    assert {'step','exp_avg','exp_avg_sq'}<=set(st)
    assert int(st['step'])==rec['successful']
    assert all(torch.isfinite(v).all().item() for v in st.values() if torch.is_tensor(v))
   compare(state['curves'],curves[:t//500]);compare(state['loss_sum']/t,rec['cumulative_loss']);compare([g['lr'] for g in state['optimizer']['param_groups']],rec['learning_rates'])
   assert state['training_seconds']>=0 and state['wall_seconds']>=state['training_seconds'];del state;gc.collect()
   metrics={}
   for role,key in [('train','train_ids'),('validation','inner_validation_ids')]:
    c,g=recount(cp/f'{role}.npz',folds[r][key],meta,truth);metric=aggregate(c,g);compare(metric,rec['source_'+role]);compare(metric,curves[t//500-1]['source_'+role]);metrics[role]=metric
   candidates[str(t)]={rr:[metrics['validation']['regions'][rr][k] for k in ('tp','fp','fn','tn')] for rr in folds[r]['training_regions']}
   gaps[str(t)]=dict(train=metrics['train'],inner=metrics['validation'],gap=metrics['train']['inner_macro_f1']-metrics['validation']['inner_macro_f1'],successful=rec['successful'],skipped=rec['skipped'],wall_seconds=rec['wall_seconds'])
   states[str(t)]=m['files']['state.pt']
  selected,scores=selection(candidates,folds[r]['training_regions']);record=read(out/'SELECTION.json')
  assert record==dict(selected_step=selected,candidates=candidates,scores=scores,protocol_sha256=identity['protocol_sha256'],config_sha256=sha(out/'config.json'))
  assert res['skipped_updates']==rec['skipped'] and res['successful_updates']==rec['successful'] and res['exposure_sha256']==rec['exposure_sha256']
  entries[run]=dict(config_sha256=sha(out/'config.json'),receipt_sha256=sha(out/'result.json'),selection_sha256=sha(out/'SELECTION.json'),selected_step=selected,states=states,gaps=gaps,skipped_updates=res['skipped_updates'])
  print('AUDITED',run,flush=True)
 for r in REG:
  for s in SEEDS:
   x=inits[r,s,'pretrained'];y=inits[r,s,'scratch'];assert x['decoder_sha256']==y['decoder_sha256'] and x['encoder_sha256']!=y['encoder_sha256']
 assert all(e==envs[0] for e in envs),'Environment mismatch'
 return dict(status='PASS',count=30,state_blobs=120,selections=30,identity=identity,environment=envs[0],runs=entries,automatic_followon=False)

def final_assessment(outer,registry,data):
 outer=Path(outer);meta,truth=source(data);counts={};metrics={};selected={};blockids={};blockcounts={}
 expected={f'{r}/seed{s}/{a}/checkpoint_{t:05d}/MANIFEST.json' for r in REG for s in SEEDS for a in ARMS for t in STEPS}
 assert {p.relative_to(outer).as_posix() for p in outer.rglob('MANIFEST.json')}==expected,'Need exactly 120 outer checkpoints'
 for r in REG:
  names=read(Path(data)/'folds'/f'{r}.json')['outer_test_ids'];keys=[(math.floor(meta[n]['center_x']/5120),math.floor(meta[n]['center_y']/5120)) for n in names];blocks=sorted(set(keys));blockids[r]=np.array([blocks.index(k) for k in keys]);blockcounts[r]=len(blocks)
  for s in SEEDS:
   for a in ARMS:
    run=f'{r}/seed{s}/{a}';entry=registry['runs'][run];selected[r,s,a]=entry['selected_step']
    for t in STEPS:
     cp=outer/run/f'checkpoint_{t:05d}';m=read(cp/'MANIFEST.json');assert m['state_sha256']==entry['states'][str(t)] and m['registry_sha256']==registry['_file_sha256']
     assert m['prediction_sha256']==sha(cp/'outer.npz')
     c,g=recount(cp/'outer.npz',names,meta,truth);counts[r,s,a,t]=c;metrics[r,s,a,t]=met(c.sum(0))
 rng=np.random.Generator(np.random.PCG64(20260928));weights={r:np.empty((2000,blockcounts[r]),np.int64) for r in REG}
 for i in range(2000):
  for r in REG:
   b=blockcounts[r];weights[r][i]=np.bincount(rng.integers(0,b,size=b),minlength=b)
 boot={}
 for (r,s,a,t),c in counts.items():
  grouped=np.zeros((blockcounts[r],4),np.int64);np.add.at(grouped,blockids[r],c);cc=weights[r]@grouped;den=2*cc[:,0]+cc[:,1]+cc[:,2];valid=(cc[:,0]+cc[:,2]>0)&(den>0);v=np.full(2000,np.nan);v[valid]=2*cc[valid,0]/den[valid];boot[r,s,a,t]=v
 summaries={};rows=[]
 for endpoint in [*STEPS,'inner_selected']:
  gains=np.empty((5,3));rep=[];absolutes={a:[] for a in ARMS}
  for ri,r in enumerate(REG):
   for si,s in enumerate(SEEDS):
    tt={a:(selected[r,s,a] if endpoint=='inner_selected' else endpoint) for a in ARMS};pp=metrics[r,s,'pretrained',tt['pretrained']];ss=metrics[r,s,'scratch',tt['scratch']];gains[ri,si]=pp['f1']-ss['f1']
    rep.append(boot[r,s,'pretrained',tt['pretrained']]-boot[r,s,'scratch',tt['scratch']]);absolutes['pretrained'].append(pp['f1']);absolutes['scratch'].append(ss['f1']);rows.append(dict(endpoint=endpoint,region=r,seed=s,steps=tt,pretrained=pp,scratch=ss,gain=float(gains[ri,si])))
  values=np.asarray(rep).reshape(5,3,2000).mean(axis=1).mean(axis=0);valid=np.isfinite(values);ci=np.quantile(values[valid],[.025,.975],method='linear').tolist() if valid.any() else None
  summaries[str(endpoint)]=dict(mean_gain=float(gains.mean(axis=1).mean()),region_gains=gains.mean(1).tolist(),seed_macro_gains=gains.mean(0).tolist(),gain_ci95=ci,valid_replicates=int(valid.sum()),absolute_macro_F1={a:float(np.mean(v)) for a,v in absolutes.items()})
 v=summaries['20000'];quality=all(b>=20 for b in blockcounts.values()) and v['valid_replicates']>=1800 and all(metrics[r,s,a,20000]['tp']>0 for r in REG for s in SEEDS for a in ARMS) and all(e['skipped_updates']<=200 for e in registry['runs'].values())
 d=decision(v['mean_gain'],v['gain_ci95'],v['region_gains'],v['seed_macro_gains'],quality)
 result=dict(status='COMPLETE_AUDITED',decision=d,primary_20000=v,endpoint_summaries=summaries,trajectory=rows,quality_sufficient=bool(quality),occupied_blocks=blockcounts,training_audit=registry,automatic_followon=False,scope='conditional on frozen models, five historical regions and three seeds; no new-event or refit inference')
 finite(result);return result

def decision(g,ci,regions,seeds,quality,technical='PASS'):
 if technical!='PASS':return None
 if not quality:return 'F4'
 if g>=.01 and ci[0]>0 and sum(x>0 for x in regions)>=4 and sum(x>0 for x in seeds)>=2:return 'F1'
 if g<=-.01 and ci[1]<0 and sum(x<0 for x in regions)>=4 and sum(x<0 for x in seeds)>=2:return 'F2'
 if abs(g)<=.005 and ci[0]>=-.01 and ci[1]<=.01 and all(abs(x)<=.01 for x in regions):return 'F3'
 return 'F4'
