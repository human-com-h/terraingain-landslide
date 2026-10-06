"""Four immutable trajectory checkpoints plus transactional rolling resume. No best selection."""
import argparse,signal,time,sys,traceback,os,shutil
from .common_g1v2 import *
from .controlled import *
from terraingain_landslide.training.resume_state import cpu_state,rng_state,restore_rng,save_resume,load_resume

def tensor_hash(module):
 import hashlib
 h=hashlib.sha256()
 for n,v in module.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def verify_initialization(out,model):
 record=dict(structure={n:list(v.shape) for n,v in model.state_dict().items()},decoder_sha256=tensor_hash(model.decoder),encoder_sha256=tensor_hash(model.encoder))
 if (out/'INITIALIZATION.json').exists():assert read(out/'INITIALIZATION.json')==record,'Initialization mismatch; preserve and audit'
 else:write(out/'INITIALIZATION.json',record)

def met(c):
 m=metrics(c);tp,fp,fn,tn=map(int,c);m['fpr']=fp/(fp+tn) if fp+tn else None;m['fnr']=fn/(fn+tp) if fn+tp else None;return m

def aggregate_full(c,g):
 m=met(c.sum(0));m['regions']={r:met(c[g==r].sum(0)) for r in sorted(set(g))}
 for endpoint in ['f1','iou','precision','recall','fpr','fnr','predicted_positive_fraction']:
  values=[m['regions'][r][endpoint] for r in sorted(set(g))];m['inner_macro_'+endpoint]=float(np.mean(values)) if all(x is not None for x in values) else None
 return m
@torch.no_grad()
def evaluate(model,x,y,ids,groups,names,fit,save=None):
 model.eval();counts=[];preds=[];truths=[]
 for start in range(0,len(ids),2):
  ii=ids[start:start+2];truth=y[ii].astype(bool)
  with amp_autocast():logits=model(batch(x,ii,fit,'cuda'))
  if not torch.isfinite(logits).all():raise RuntimeError('nonfinite evaluation logits')
  pred=(logits.float().sigmoid()>=.5).cpu().numpy();counts.append(np.column_stack([(pred&truth).sum((1,2)),(pred&~truth).sum((1,2)),(~pred&truth).sum((1,2)),(~pred&~truth).sum((1,2))]))
  if save is not None:preds.append(pred);truths.append(truth)
 c=np.concatenate(counts);m=aggregate_full(c,groups[ids])
 if save is not None:
  pp=np.concatenate(preds);tt=np.concatenate(truths);np.savez_compressed(save,shape=np.array(pp.shape),prediction=np.packbits(pp.reshape(len(pp),-1),axis=1),truth=np.packbits(tt.reshape(len(tt),-1),axis=1),counts=c,names=np.array(names)[ids],groups=groups[ids])
 return m

def commit_checkpoint(out,step,state,stage):
 dest=out/f'checkpoint_{step:05d}'
 if dest.exists():
  prior=read(dest/'MANIFEST.json');assert prior['step']==step and prior['config_sha256']==state['config_sha256']
  for n,h in prior['files'].items():assert sha(dest/n)==h
  return dest
 with (stage/'state.pt').open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
 write(stage/'checkpoint.json',dict(step=step,primary=step==20000,config_sha256=state['config_sha256'],environment=state['environment'],successful=step-state['skipped'],skipped=state['skipped'],cumulative_loss=state['loss_sum']/step,loss_sum=state['loss_sum'],learning_rates=[g['lr'] for g in state['optimizer']['param_groups']],exposure_sha256=state['exposure_sha256'],source_train=state['curves'][-1]['source_train'],source_validation=state['curves'][-1]['source_validation'],wall_seconds=state['wall_seconds']))
 write(stage/'MANIFEST.json',dict(step=step,config_sha256=state['config_sha256'],files={f.name:sha(f) for f in stage.iterdir() if f.is_file()}));stage.replace(dest);return dest

def recover_committed(out,config_hash,steps):
 state=load_resume(out,config_hash)
 # Recover a checkpoint committed before its rolling pointer, without replaying its evaluation.
 for t in steps:
  cp=out/f'checkpoint_{t:05d}'
  if cp.exists() and t>=state['step']:
   manifest=read(cp/'MANIFEST.json');assert manifest['config_sha256']==config_hash
   for n,h in manifest['files'].items():assert sha(cp/n)==h
   state=torch.load(cp/'state.pt',map_location='cpu',weights_only=False)
 assert state['config_sha256']==config_hash
 return state

def train(args):
 protocol=training_protocol();authorize(args.authorization,args.split,args.seed);verify_package()
 assert args.arm in protocol['arms'];assert torch.__version__==protocol['environment']['kaggle_torch'];e=environment();assert e['torchvision']==protocol['environment']['kaggle_torchvision'] and e['gpu']=='Tesla T4'
 x,y,tr,va,groups,names,fit=load_data(data_root(),args.split)
 out=args.output/args.split/f'seed{args.seed}'/args.arm;out.mkdir(parents=True,exist_ok=True)
 cfg=dict(version=protocol['version'],split=args.split,seed=args.seed,arm=args.arm,protocol_sha256=sha(P/'S05F_PROTOCOL.json'),package_sha256=sha(P/'PACKAGE_MANIFEST.json'),input_manifest_sha256=sha(P/'INPUT_MANIFEST.json'),split_sha256=sha(data_root()/'FOLD.json'),fit_sha256=sha(data_root()/'FIT.json'),data_manifest_sha256=sha(data_root()/'DATA_MANIFEST.json'),environment_identity=env_identity(e),environment=e,authorization_sha256=sha(args.authorization))
 if (out/'config.json').exists():
  old=read(out/'config.json')
  for k in ['split','seed','arm','protocol_sha256','package_sha256','input_manifest_sha256','split_sha256','fit_sha256','data_manifest_sha256','environment_identity']:assert old[k]==cfg[k],k
  if (out/'result.json').exists():
   done=read(out/'result.json');assert done['primary_step']==20000
   for n,h in done['files'].items():assert sha(out/n)==h
   return 'COMPLETE'
  assert (out/'resume_pointer.json').exists(),'Incomplete run without committed recovery; preserve and investigate'
 else:write(out/'config.json',cfg)
 model=model_for(args.arm,args.seed).cuda();verify_initialization(out,model);opt=optimizer_for(model,protocol);scaler=amp_scaler(init_scale=protocol['amp']['initial_scale'],growth_interval=protocol['amp']['growth_interval']);sampler=torch.Generator().manual_seed(args.seed+1000);aug=torch.Generator().manual_seed(args.seed+3000);weight=torch.tensor(fit['positive_weight'],device='cuda');positive=tr[y[tr].sum((1,2))>0];negative=tr[y[tr].sum((1,2))==0]
 step=0;curves=[];losses=[];loss_sum=0.;skipped=0;chain='00'*32;train_seconds=0.;prior_wall=0.;started=time.time();stop=[False]
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda signum,frame:stop.__setitem__(0,True))
 if (out/'resume_pointer.json').exists():
  state=recover_committed(out,sha(out/'config.json'),protocol['checkpoints'])
  assert state['config_sha256']==sha(out/'config.json') and env_identity(state['environment'])==env_identity(e);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);restore_rng(state['rng'],sampler,aug)
  step=state['step'];curves=state['curves'];losses=state['losses'];loss_sum=state['loss_sum'];skipped=state['skipped'];chain=state['exposure_sha256'];train_seconds=state['training_seconds'];prior_wall=state['wall_seconds'];del state
 def snapshot():return dict(config_sha256=sha(out/'config.json'),model=cpu_state(model),optimizer=opt.state_dict(),scaler=scaler.state_dict(),rng=rng_state(sampler,aug),step=step,curves=curves,losses=losses,loss_sum=loss_sum,skipped=skipped,exposure_sha256=chain,training_seconds=train_seconds,wall_seconds=prior_wall+time.time()-started,environment=environment())
 if not (out/'resume_pointer.json').exists():save_resume(out,snapshot())
 write(out/f'session_{time.time_ns()}.json',dict(resumed_at=step,environment=e))
 for step in range(step+1,protocol['attempts']+1):
  torch.cuda.synchronize();tick=time.perf_counter();ids,k,flip,chain=exposure(positive,negative,sampler,aug,chain);a=batch(x,ids,fit,'cuda');b=torch.tensor(y[ids].astype(np.float32),device='cuda');a=torch.rot90(a,k,(-2,-1));b=torch.rot90(b,k,(-2,-1))
  if flip:a=a.flip(-1);b=b.flip(-1)
  loss,skip=update(model,opt,scaler,a,b,weight,step,protocol);skipped+=skip;loss_sum+=loss;losses.append(loss);losses=losses[-500:];torch.cuda.synchronize();train_seconds+=time.perf_counter()-tick
  if step%protocol['eval_interval']==0:
   stage=out/f'.checkpoint_{step:05d}_{time.time_ns()}.tmp' if step in protocol['checkpoints'] else None
   if stage is not None:stage.mkdir()
   tm=evaluate(model,x,y,tr,groups,names,fit,stage/'train.npz' if stage else None);vm=evaluate(model,x,y,va,groups,names,fit,stage/'validation.npz' if stage else None)
   curves.append(dict(attempt=step,successful_updates=step-skipped,skipped_amp_updates=skipped,loss_mean_last_interval=float(np.mean(losses)),cumulative_loss=loss_sum/step,learning_rates=[g['lr'] for g in opt.param_groups],source_train=tm,source_validation=vm,exposure_sha256=chain,wall_seconds=prior_wall+time.time()-started));write(out/'curve.json',curves)
   if stage is not None:
    assert chain==read(P/'EXPECTED_EXPOSURE_CHAINS.json')[f'{args.split}/seed{args.seed}'][str(step)]
    commit_checkpoint(out,step,snapshot(),stage)
   print(args.split,args.seed,args.arm,step,vm['inner_macro_f1'],flush=True)
  pause=stop[0] or time.time()>=args.deadline
  if step%protocol['recovery_interval']==0 or pause:save_resume(out,snapshot())
  if pause and step<protocol['attempts']:
   write(out/'PAUSED.json',dict(step=step,scientific_stop=False));return 'PAUSED'
 assert step==20000 and len(curves)==40
 files={}
 for t in protocol['checkpoints']:
  cp=out/f'checkpoint_{t:05d}';man=read(cp/'MANIFEST.json');assert man['step']==t
  for n,h in man['files'].items():assert sha(cp/n)==h;files[f'checkpoint_{t:05d}/{n}']=h
  files[f'checkpoint_{t:05d}/MANIFEST.json']=sha(cp/'MANIFEST.json')
 from terraingain_landslide.selection.protocol_rules import select_inner
 candidates={t:{r:[read(out/f'checkpoint_{t:05d}'/'checkpoint.json')['source_validation']['regions'][r][k] for k in ('tp','fp','fn','tn')] for r in read(data_root()/'FOLD.json')['training_regions']} for t in protocol['checkpoints']}
 selected,scores=select_inner(candidates,read(data_root()/'FOLD.json')['training_regions'])
 write(out/'SELECTION.json',dict(selected_step=selected,candidates=candidates,scores=scores,protocol_sha256=sha(P/'S05F_PROTOCOL.json'),config_sha256=sha(out/'config.json')))
 files['SELECTION.json']=sha(out/'SELECTION.json')
 files['INITIALIZATION.json']=sha(out/'INITIALIZATION.json');files['config.json']=sha(out/'config.json');files['curve.json']=sha(out/'curve.json');write(out/'result.json',dict(status='COMPLETE',primary_step=20000,checkpoint_set=protocol['checkpoints'],files=files,successful_updates=20000-skipped,skipped_updates=skipped,exposure_sha256=chain))
 # Final full state is retained in immutable checkpoint_20000; rolling duplicate no longer needed.
 from terraingain_landslide.training.resume_state import clean_completed_resume
 clean_completed_resume(out);(out/'PAUSED.json').unlink(missing_ok=True)
 return 'COMPLETE'

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=REGIONS,required=True);p.add_argument('--seed',type=int,choices=[17,29,43],required=True);p.add_argument('--arm',choices=['pretrained','scratch'],required=True);p.add_argument('--authorization',type=Path,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
 try:return 75 if train(a)=='PAUSED' else 0
 except BaseException as e:write(a.output/f'failure_{a.split}_{a.seed}_{a.arm}_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':sys.exit(main())
