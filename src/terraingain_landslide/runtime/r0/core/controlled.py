"""S05E frozen architecture/optimization primitives. Imports never train."""
import os,random,math,hashlib
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np,torch
from torch import nn
from .common_g1v2 import *
from terraingain_landslide.models.model_g1v2 import Segmenter,load_encoder,full_batch_logits,loss_fn,interpolate_position
from terraingain_landslide.models.official_pos_embed import get_2d_sincos_pos_embed,get_1d_sincos_pos_embed_from_grid
from terraingain_landslide.training.amp_compat import amp_scaler,amp_autocast

def settings(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
def model_for(arm,seed):
 assert arm in ['pretrained','scratch'];settings(seed);model=Segmenter()
 # Decoder construction and RNG position are identical across arms.
 if arm=='pretrained':load_encoder(model,encoder_path())
 else:
  for module in model.encoder.modules():
   if isinstance(module,nn.Linear):nn.init.trunc_normal_(module.weight,std=.02);nn.init.zeros_(module.bias)
   elif isinstance(module,nn.LayerNorm):nn.init.ones_(module.weight);nn.init.zeros_(module.bias)
  nn.init.trunc_normal_(model.encoder.cls_token,std=.02)
  with torch.no_grad():
   pos=torch.from_numpy(get_2d_sincos_pos_embed(512,12,cls_token=True)).float()[None]
   model.encoder.pos_embed.copy_(interpolate_position(pos));model.encoder.channel_embed.copy_(torch.from_numpy(get_1d_sincos_pos_embed_from_grid(256,np.arange(3))).float()[None]);model.encoder.channel_cls_embed.zero_()
 return model

def optimizer_for(model,p):
 groups=[]
 for role,module in [('encoder',model.encoder),('decoder',model.decoder)]:
  for decay in [False,True]:
   values=[v for n,v in module.named_parameters() if v.requires_grad and (v.ndim>1 and n not in {'pos_embed','channel_embed','channel_cls_embed','cls_token'})==decay]
   if values:groups.append(dict(params=values,lr=p['optimization'][role+'_lr'],initial_lr=p['optimization'][role+'_lr'],weight_decay=p['optimization']['weight_decay'] if decay else 0.,role=role))
 return torch.optim.AdamW(groups,betas=tuple(p['optimization']['betas']),eps=p['optimization']['eps'])
def schedule(step,p):
 warm=p['optimization']['warmup_attempts'];floor=p['optimization']['final_lr_ratio'];return step/warm if step<=warm else floor+(1-floor)*(1+math.cos(math.pi*(step-warm)/(p['attempts']-warm)))/2

def exposure(positive,negative,sampler,aug,chain):
 ids=np.r_[positive[torch.randint(len(positive),(4,),generator=sampler).numpy()],negative[torch.randint(len(negative),(4,),generator=sampler).numpy()]].astype(np.int64)
 k=int(torch.randint(4,(1,),generator=aug));flip=bool(torch.randint(2,(1,),generator=aug));chain=hashlib.sha256(bytes.fromhex(chain)+ids.astype('<i8').tobytes()+bytes([k,int(flip)])).hexdigest();return ids,k,flip,chain

def update(model,opt,scaler,a,b,weight,step,p):
 model.train()
 for g in opt.param_groups:g['lr']=g['initial_lr']*schedule(step,p)
 opt.zero_grad(set_to_none=True)
 with amp_autocast():logits=full_batch_logits(model,a,p['microbatch'],True);loss=loss_fn(logits,b,weight)
 if not torch.isfinite(loss) or not torch.isfinite(logits).all():raise RuntimeError('nonfinite loss/logits')
 scaler.scale(loss).backward();old=scaler.get_scale();scaler.step(opt);scaler.update()
 return float(loss.detach()),int(scaler.get_scale()<old)

def env_identity(e):return {k:e[k] for k in ['torch','torchvision','numpy','cuda','cudnn','gpu']}
def verify_package():
 from terraingain_landslide.utils.integrity import verify_source_tree
 verify_source_tree()
 for n,h in read(P/'PACKAGE_MANIFEST.json')['files'].items():
  assert sha(P/n)==h,'Package mismatch: '+n

def authorize(path,split,seed):
 a=read(path);assert a['explicit_user_authorization'] is True and a['protocol_sha256']==sha(P/'S05F_PROTOCOL.json');assert [split,seed] in a['allowed_pairs'];assert a['max_formal_runs']==30;assert a.get('scientific_count_before')==607
 return a


def encoder_path():
 from terraingain_landslide.utils.integrity import checked_encoder
 return checked_encoder(os.environ['TERRAINGAIN_ENCODER'])
