from pathlib import Path
import json,hashlib,os,platform,sys
import numpy as np
import torch
P=Path(__file__).resolve().parent
REGIONS=['dominicamaria','italy','hiroshima','hokkaido','thrissur']
COMMON=REGIONS
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);q=p.with_suffix(p.suffix+'.tmp')
 q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def metrics(count):
 tp,fp,fn,tn=map(int,count);den=2*tp+fp+fn;total=tp+fp+fn+tn
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,f1=2*tp/den if den else None,iou=tp/(tp+fp+fn) if tp+fp+fn else None,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,predicted_positive_fraction=(tp+fp)/total,reference_positive_fraction=(tp+fn)/total,all_background_prediction=tp+fp==0)
def aggregate(counts,groups):
 result=metrics(np.asarray(counts).sum(0))
 result['regions']={r:metrics(np.asarray(counts)[np.asarray(groups)==r].sum(0)) for r in sorted(set(groups))}
 result['common_four_macro_f1']=float(np.mean([result['regions'][r]['f1'] for r in COMMON]))
 return result
def batch(x,ids,fit,device):
 a=x[ids].astype(np.float32)
 a=(a-np.asarray(fit['mean'],np.float32)[None,:,None,None])/np.asarray(fit['std'],np.float32)[None,:,None,None]
 return torch.from_numpy(a).to(device)
def environment():
 import torchvision
 return dict(torchvision=torchvision.__version__,python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),command=sys.argv)


def training_protocol():
 p=read(P/'S05F_PROTOCOL.json')
 return dict(p,**p['recipe'],eval_interval=p['inner_eval_interval'])

def data_root():return Path(os.environ['S05F_DATA'])

def verify_data(held=None,outer=False):
 d=data_root();m=read(d/'DATA_MANIFEST.json');catalog=read(P/'SOURCE_PACKS.json');key='outer' if outer else held
 assert sha(d/'DATA_MANIFEST.json')==catalog[key], 'Wrong data package'
 assert m['kind']==('outer' if outer else 'training')
 if not outer:assert m['held_out']==held
 actual={p.relative_to(d).as_posix() for p in d.rglob('*') if p.is_file()}
 assert actual==set(m['files'])|{'DATA_MANIFEST.json'}, 'Extra/missing data files'
 for n,h in m['files'].items():assert sha(d/n)==h,n
 return d

def load_data(data,split_name):
 from terraingain_landslide.data.fold_contract import fit_train_only,validate
 d=verify_data(split_name);fold=read(d/'FOLD.json');validate(fold);frozen=read(d/'FIT.json');cache={}
 def loader(r):
  assert r!=split_name
  if r not in cache:
   rec=read(d/'source_optical'/f'{r}.json')['records']
   with np.load(d/'source_optical'/f'{r}.npz',allow_pickle=False) as z:x=z['x'];y=z['y']
   assert x.shape==(256,10,128,128) and y.shape==(256,128,128)
   assert np.isfinite(x).all() and np.isin(y,[0,1]).all()
   assert np.array_equal(y.sum((1,2)),[v['positive_pixels'] for v in rec])
   cache[r]=(x,y,[v['name'] for v in rec])
  return cache[r]
 fit=fit_train_only(fold,loader);assert fit==frozen,'Train-only fit mismatch'
 names=fold['train_ids']+fold['inner_validation_ids'];lookup={n:(r,i) for r in fold['training_regions'] for i,n in enumerate(loader(r)[2])}
 x=np.stack([cache[lookup[n][0]][0][lookup[n][1]] for n in names]);y=np.stack([cache[lookup[n][0]][1][lookup[n][1]] for n in names])
 n=len(fold['train_ids']);return x,y,np.arange(n),np.arange(n,len(names)),np.array([lookup[v][0] for v in names]),names,fit
