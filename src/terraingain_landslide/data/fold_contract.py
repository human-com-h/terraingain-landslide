"""S05F manifest/fit contract only. No model construction, optimizer or training entrypoint."""
import hashlib,json
import numpy as np
REGIONS=('dominicamaria','italy','hiroshima','hokkaido','thrissur')
def fingerprint(names):return hashlib.sha256(json.dumps(names,separators=(',',':')).encode()).hexdigest()
def validate(fold):
 held=fold['held_out'];assert held in REGIONS
 assert fold['training_regions']==[r for r in REGIONS if r!=held]
 groups={role:fold[role+'_ids'] for role in ('train','inner_validation','outer_test','excluded')}
 for names in groups.values():assert len(names)==len(set(names))
 for i,(a,aa) in enumerate(groups.items()):
  for b,bb in list(groups.items())[i+1:]:assert not set(aa)&set(bb),(a,b)
 assert all(n.startswith(held+'_') for n in groups['outer_test'])
 assert all(not n.startswith(held+'_') for role in ('train','inner_validation','excluded') for n in groups[role])
 assert fingerprint(groups['train'])==fold['training_ids_sha256']
 return True

def fit_train_only(fold,loader):
 validate(fold);wanted=set(fold['train_ids']);sm=np.zeros(10);sq=np.zeros(10);pixels=0;positive=0;used=set();pos=[];neg=[]
 for region in fold['training_regions']:
  # By contract the loader never receives the outer region.
  x,y,names=loader(region)
  for i,n in enumerate(names):
   if n not in wanted:continue
   a=x[i].astype(np.float64);b=y[i];assert np.isfinite(a).all() and np.isin(b,[0,1]).all()
   sm+=a.sum((1,2));sq+=(a*a).sum((1,2));pixels+=b.size;cnt=int(b.sum());positive+=cnt;used.add(n)
   (pos if cnt>0 else neg).append(n)
 assert used==wanted and pos and neg and 0<positive<pixels
 mu=sm/pixels;sd=np.sqrt(np.maximum(sq/pixels-mu*mu,0)).clip(.0001)
 order={n:i for i,n in enumerate(fold['train_ids'])}
 return dict(held_out=fold['held_out'],fit_names=fold['train_ids'],fit_names_sha256=fingerprint(fold['train_ids']),mean=mu.tolist(),std=sd.tolist(),positive_weight=float(np.clip(np.sqrt((pixels-positive)/positive),1,20)),fit_pixels=pixels,positive_pixels=positive,positive_indices=sorted(order[n] for n in pos),negative_indices=sorted(order[n] for n in neg),outer_data_used=False,validation_data_used=False)
