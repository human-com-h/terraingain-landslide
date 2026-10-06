"""Executable protocol assertions only; no training or assessment on real model outputs."""
from fractions import Fraction
STEPS=(5000,10000,15000,20000)
def select_inner(candidates,regions):
 assert set(candidates)==set(STEPS) and len(regions)==4 and len(set(regions))==4
 scores={}
 for step in STEPS:
  assert set(candidates[step])==set(regions)
  parts=[]
  for region in regions:
   tp,fp,fn,tn=candidates[step][region];assert all(isinstance(n,int) and n>=0 for n in (tp,fp,fn,tn))
   den=2*tp+fp+fn;parts.append(Fraction(2*tp,den) if den else Fraction(0))
  scores[step]=sum(parts)/4
 best=max(scores.values());selected=min(t for t in STEPS if scores[t]==best)
 return selected,{str(t):dict(numerator=x.numerator,denominator=x.denominator) for t,x in scores.items()}
def decide(primary,ci,region_gains,seed_macro_gains,quality,technical_status='PASS'):
 if technical_status!='PASS':return None
 assert len(region_gains)==5 and len(seed_macro_gains)==3
 if not quality:return 'F4'
 if primary>=.01 and ci[0]>0 and sum(x>0 for x in region_gains)>=4 and sum(x>0 for x in seed_macro_gains)>=2:return 'F1'
 if primary<=-.01 and ci[1]<0 and sum(x<0 for x in region_gains)>=4 and sum(x<0 for x in seed_macro_gains)>=2:return 'F2'
 if abs(primary)<=.005 and ci[0]>=-.01 and ci[1]<=.01 and all(abs(x)<=.01 for x in region_gains):return 'F3'
 return 'F4'
