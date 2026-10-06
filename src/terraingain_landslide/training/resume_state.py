"""Transactional recovery state; only load locally produced, hash-verified state."""
import os,random,time
import numpy as np
import torch
from terraingain_landslide.utils.io import *

def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def rng_state(sampler,aug):
 return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],sampler=sampler.get_state(),augmentation=aug.get_state())
def restore_rng(r,sampler,aug):
 random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
 if r['cuda']:torch.cuda.set_rng_state_all(r['cuda'])
 sampler.set_state(r['sampler']);aug.set_state(r['augmentation'])
def save_resume(out,state):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 old=read(out/'resume_pointer.json') if (out/'resume_pointer.json').exists() else None
 name=f'resume_{state["step"]:05d}_{time.time_ns()}.pt';path=out/name;tmp=out/(name+'.tmp')
 with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
 tmp.replace(path)
 write(out/'resume_pointer.json',dict(file=name,sha256=sha(path),step=state['step'],config_sha256=state['config_sha256']))
 # Remove only the previously committed file, after the new pointer is committed.
 if old:
  prior=out/old['file'];assert prior.parent.resolve()==out.resolve() and prior.name.startswith('resume_') and prior.suffix=='.pt'
  if prior!=path and prior.exists():prior.unlink()
 return name

def load_resume(out,config_sha):
 out=Path(out);r=read(out/'resume_pointer.json');path=out/r['file']
 assert path.parent.resolve()==out.resolve() and path.suffix=='.pt'
 assert r['config_sha256']==config_sha and sha(path)==r['sha256'],'Recovery integrity mismatch'
 state=torch.load(path,map_location='cpu',weights_only=False)
 assert state['config_sha256']==config_sha and state['step']==r['step']
 return state

def clean_completed_resume(out):
 out=Path(out)
 if (out/'resume_pointer.json').exists():
  r=read(out/'resume_pointer.json');p=out/r['file'];assert p.parent.resolve()==out.resolve() and p.name.startswith('resume_') and p.suffix=='.pt'
  p.unlink(missing_ok=True);(out/'resume_pointer.json').unlink()
