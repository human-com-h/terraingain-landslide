"""Exact source-only selection between R0/R1 fixed-20k candidates. No model imports."""
import json,hashlib,os
from pathlib import Path
from fractions import Fraction
CORE=Path(__file__).resolve().parents[1]/'runtime'/'r1'/'core'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def score(regional_counts,regions):
 if set(regional_counts)!=set(regions) or len(regions)!=4:raise ValueError('Wrong source-region selection support')
 parts=[]
 for region in regions:
  values=regional_counts[region]
  if len(values)!=4 or any(type(x) is not int or x<0 for x in values):raise ValueError('Invalid confusion counts')
  tp,fp,fn,tn=values;den=2*tp+fp+fn
  parts.append(Fraction(2*tp,den) if den else Fraction(0))
 return sum(parts)/4

def select(r0,r1,regions):
 first,second=score(r0,regions),score(r1,regions)
 return ('R1' if second>first else 'R0'),{
  'R0':dict(numerator=first.numerator,denominator=first.denominator),
  'R1':dict(numerator=second.numerator,denominator=second.denominator)}

def seal_selection(root,registry,output):
 p=read(CORE/'S05G_PROTOCOL.json');reference=Path(os.environ['S05G_R0_REFERENCE'])/'R0_SOURCE_20K_COUNTS.json'
 if sha(reference)!=p['parent']['R0_SOURCE_20K_COUNTS_sha256']:raise RuntimeError('R0 source-count reference mismatch')
 original=read(reference)
 if original['contains_outer_scores'] is not False:raise RuntimeError('Outer scores forbidden in source selection')
 result={}
 for run in sorted(registry['runs']):
  held,seed,arm=run.split('/');regions=[r for r in p['regions'] if r!=held]
  selection=read(Path(root)/run/'SELECTION.json')
  selected,scores=select(original['counts'][run],selection['candidates']['20000'],regions)
  result[run]=dict(selected_recipe=selected,checkpoint=20000,scores_exact=scores,
   source_regions=regions,r1_source_selection_sha256=sha(Path(root)/run/'SELECTION.json'))
 if len(result)!=30:raise RuntimeError('All 30 source choices required before Outer')
 sealed=dict(status='SOURCE_ONLY_RECIPE_SELECTION_SEALED',protocol_sha256=sha(CORE/'S05G_PROTOCOL.json'),
  r1_registry_sha256=sha(Path(output).parent/'TRAINING_REGISTRY.json'),r0_protocol_sha256=p['parent']['S05F_protocol_sha256'],
  r0_source_counts_sha256=sha(reference),checkpoint=20000,tie_break='R0',nominal_candidate_attempts_per_arm=40000,
  outer_used_for_selection=False,records=result)
 text=json.dumps(sealed,ensure_ascii=False,indent=2)+'\n';output=Path(output)
 if output.exists() and output.read_text(encoding='utf-8')!=text:raise RuntimeError('Source choices changed after sealing')
 if not output.exists():output.write_text(text,encoding='utf-8')
 return sealed
