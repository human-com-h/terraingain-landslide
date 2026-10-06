"""Synthetic-only engineering tests. No source arrays, no scientific runs."""
import argparse,tempfile,copy,time,gc,json,hashlib
from pathlib import Path
import numpy as np,torch
from .common_g1v2 import P,read,write,sha,environment
from .controlled import model_for,optimizer_for,settings,update,exposure,amp_scaler,env_identity
from terraingain_landslide.models.model_g1v2 import full_batch_logits,loss_fn
from terraingain_landslide.training.resume_state import cpu_state,rng_state,restore_rng,save_resume,load_resume
from .train_s05f import evaluate,commit_checkpoint,tensor_hash
from .assess_s05f import met,compare

def run(preflight=False):
 from .common_g1v2 import training_protocol
 p=training_protocol();settings(17);m=model_for('pretrained',17);structure={n:list(v.shape) for n,v in m.state_dict().items()};decoder=tensor_hash(m.decoder);enc=tensor_hash(m.encoder);pos=m.encoder.pos_embed.detach().clone();channel=m.encoder.channel_embed.detach().clone();params=sum(v.numel() for v in m.parameters());del m
 m=model_for('scratch',17);assert structure=={n:list(v.shape) for n,v in m.state_dict().items()};assert decoder==tensor_hash(m.decoder) and enc!=tensor_hash(m.encoder);assert torch.equal(pos,m.encoder.pos_embed) and torch.equal(channel,m.encoder.channel_embed);assert torch.count_nonzero(m.encoder.pos_embed)>0;del m
 write(P/'MODEL_STRUCTURE.json',structure) if not preflight else None
 # Full 20k exposure generation only: no model forward or optimizer.
 traces=[]
 for arm in ['pretrained','scratch']:
  sa=torch.Generator().manual_seed(1017);au=torch.Generator().manual_seed(3017);chain='00'*32;trace=[]
  for step in range(1,20001):
   ids,k,f,chain=exposure(np.arange(10),np.arange(10,30),sa,au,chain)
   if step in p['checkpoints']:trace.append(chain)
  traces.append(trace)
 assert traces[0]==traces[1]
 # CPU nonseparable global Dice gradient equivalence.
 a=torch.randn(8,10,8,8);b=(torch.rand(8,8,8)>.7).float();toy=torch.nn.Sequential(torch.nn.Conv2d(10,1,1),torch.nn.Flatten(0,0))
 class Toy(torch.nn.Module):
  def __init__(self):super().__init__();self.c=torch.nn.Conv2d(10,1,1)
  def forward(self,x):return self.c(x)[:,0]
 toy=Toy();toy2=copy.deepcopy(toy);loss_fn(toy(a),b,torch.tensor(2.)).backward();loss_fn(full_batch_logits(toy2,a,2,True),b,torch.tensor(2.)).backward();err=max(float((u.grad-v.grad).abs().max()) for u,v in zip(toy.parameters(),toy2.parameters()));assert err<1e-6
 assert torch.cuda.is_available();memory={};resume={};initial_skip={}
 for arm in ['pretrained','scratch']:
  settings(17);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();m=model_for(arm,17).cuda();opt=optimizer_for(m,p);sc=amp_scaler(init_scale=256);sam=torch.Generator().manual_seed(1017);aug=torch.Generator().manual_seed(3017)
  gen=torch.Generator().manual_seed(901);x=torch.randn(8,10,128,128,generator=gen).cuda();y=(torch.rand(8,128,128,generator=gen)>.97).float().cuda();w=torch.tensor(5.,device='cuda');chain='00'*32;before=m.encoder.patch_embed[0].proj.weight.detach().clone();trace=[];start=time.time()
  with tempfile.TemporaryDirectory(prefix='s05e_synthetic_') as tmp:
   out=Path(tmp)
   for step in [1,2]:
    ids,k,f,chain=exposure(np.arange(4),np.arange(4,8),sam,aug,chain);aa=torch.rot90(x[ids],k,(-2,-1));bb=torch.rot90(y[ids],k,(-2,-1))
    if f:aa=aa.flip(-1);bb=bb.flip(-1)
    loss,skip=update(m,opt,sc,aa,bb,w,step,p);assert skip==0 and np.isfinite(loss);trace.append((loss,chain));torch.cuda.synchronize()
    if step==1:save_resume(out,dict(step=1,config_sha256='synthetic',model=cpu_state(m),optimizer=opt.state_dict(),scaler=sc.state_dict(),rng=rng_state(sam,aug),exposure_sha256=chain))
   after=cpu_state(m);afteropt=copy.deepcopy(opt.state_dict());after_scale=sc.state_dict();assert not torch.equal(before,m.encoder.patch_embed[0].proj.weight)
   memory[arm]=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),seconds_two_steps=time.time()-start,successful_steps=2,scale=256)
   assert memory[arm]['reserved']<12*1024**3
   del m,opt,sc;gc.collect();torch.cuda.empty_cache()
   m=model_for(arm,17).cuda();opt=optimizer_for(m,p);sc=amp_scaler(init_scale=256);state=load_resume(out,'synthetic');m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sc.load_state_dict(state['scaler']);restore_rng(state['rng'],sam,aug);chain=state['exposure_sha256'];del state
   ids,k,f,chain=exposure(np.arange(4),np.arange(4,8),sam,aug,chain);aa=torch.rot90(x[ids],k,(-2,-1));bb=torch.rot90(y[ids],k,(-2,-1))
   if f:aa=aa.flip(-1);bb=bb.flip(-1)
   loss,skip=update(m,opt,sc,aa,bb,w,2,p);actual=cpu_state(m);maxerr=max(float((v-after[n]).abs().max()) for n,v in actual.items());assert maxerr==0. and loss==trace[1][0] and chain==trace[1][1] and sc.state_dict()==after_scale
   optimizer_error=0.;optimizer_exact=True
   for k,v in opt.state_dict()['state'].items():
    for n,t in v.items():
     ref=afteropt['state'][k][n];optimizer_error=max(optimizer_error,float((t-ref).abs().max()));optimizer_exact &= torch.equal(t,ref)
     assert torch.equal(t,ref),(n,float((t-ref).abs().max()))
   resume[arm]=dict(max_weight_difference=maxerr,identical_loss=True,optimizer_bitwise_equal=optimizer_exact,optimizer_max_difference=optimizer_error,optimizer_allclose_rtol=0.,optimizer_allclose_atol=0.,identical_scaler=True,identical_exposure=True)
   # Full prediction export and independent recount on synthetic pixels only.
   xx=x.cpu().numpy();yy=y.cpu().numpy();groups=np.array(['dominicamaria','italy','hiroshima','thrissur']*2);names=[f'synthetic_{i}' for i in range(8)];fit=dict(mean=[0.]*10,std=[1.]*10)
   metric=evaluate(m,xx,yy,np.arange(8),groups,names,fit,out/'pred.npz')
   with np.load(out/'pred.npz') as z:
    pred=np.unpackbits(z['prediction'],axis=1,count=16384).reshape(8,128,128).astype(bool);truth=np.unpackbits(z['truth'],axis=1,count=16384).reshape(8,128,128).astype(bool);counts=np.column_stack([(pred&truth).sum((1,2)),(pred&~truth).sum((1,2)),(~pred&truth).sum((1,2)),(~pred&~truth).sum((1,2))]);assert np.array_equal(counts,z['counts']);compare(met(counts.sum(0)),metric)
   pointer=read(out/'resume_pointer.json');(out/'resume_orphan.pt.tmp').write_bytes(b'uncommitted');assert load_resume(out,'synthetic')['step']==1
   try:load_resume(out,'wrong')
   except AssertionError:pass
   else:raise AssertionError('Wrong config accepted')
   with (out/pointer['file']).open('ab') as f:f.write(b'corrupt')
   try:load_resume(out,'synthetic')
   except AssertionError:pass
   else:raise AssertionError('Corrupted state accepted')
   del m,opt,sc,x,y,aa,bb,actual,after,afteropt;gc.collect();torch.cuda.empty_cache()
 return dict(status='PASS',synthetic_only=True,scientific_training_increment=0,scientific_count=607,environment=environment(),parameter_count=params,architecture_identity=True,decoder_initialization_identity=True,scratch_position_channel_equal_official_fixed_geometry=True,paired_20000_exposure_hashes=traces[0],global_dice_gradient_max_difference=err,memory=memory,resume=resume,prediction_recount=True,corruption_rejected=True,wrong_config_rejected=True,orphan_ignored=True,branch_tests=False,twenty_k_primary_immutable=p['primary_checkpoint']==20000 and p['checkpoints']==[5000,10000,15000,20000])
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--preflight',action='store_true');a.add_argument('--output',type=Path,required=True);args=a.parse_args();result=run(args.preflight);write(args.output,result);print(json.dumps(result,indent=2))
