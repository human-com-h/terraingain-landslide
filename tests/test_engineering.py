"""CPU metadata checks; no model imports, arrays, states, or result analysis."""
from pathlib import Path
import ast, copy, importlib, json, subprocess, sys, tempfile, unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from terraingain_landslide.configuration import paths, protocol
from terraingain_landslide.data.fold_contract import validate
from terraingain_landslide.metadata import recovery_pointer
from terraingain_landslide.selection.bounded_selection import select

class Engineering(unittest.TestCase):
    def fixture(self, name):
        return json.loads((ROOT/'tests/fixtures'/name).read_text(encoding='utf-8'))

    def test_configuration_identity(self):
        for recipe, rate in [('r0',6.25e-6), ('r1',3.125e-5)]:
            p=protocol(recipe)
            self.assertEqual(p['recipe']['optimization']['encoder_lr'],rate)
            self.assertEqual(p['recipe']['optimization']['decoder_lr'],0.001)
            self.assertEqual(p['checkpoints'],[5000,10000,15000,20000])
            self.assertFalse(p['formal_training_authorized'])
            public=json.loads((ROOT/'configs'/(recipe+'.yaml')).read_text())
            self.assertEqual(public,p)

    def test_paths_follow_config_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'paths.yaml'
            file.write_bytes((ROOT/'configs/paths.example.yaml').read_bytes())
            p=paths(file)
            self.assertEqual(Path(p['training_data']), (Path(tmp)/'../inputs/fold_training').resolve())
            self.assertEqual(p['previous_outputs'],[])

    def test_fold_metadata_only(self):
        fold=self.fixture('fold.json')
        self.assertTrue(validate(fold))
        for role in ['train','inner_validation','excluded']:
            broken=copy.deepcopy(fold)
            broken[role+'_ids'].append(fold['outer_test_ids'][0])
            with self.assertRaises(AssertionError):validate(broken)
        broken=copy.deepcopy(fold);broken['training_ids_sha256']='changed'
        with self.assertRaises(AssertionError):validate(broken)

    def test_exact_selection_metadata_fixture(self):
        f=self.fixture('selection.json');regions=f['regions']
        choice,scores=select(f['r0'],f['r1'],regions)
        self.assertEqual(choice,'R0')
        self.assertEqual(scores['R0'],scores['R1'])
        zeros={r:[0,0,0,8] for r in regions}
        self.assertEqual(select(zeros,zeros,regions)[1]['R0'],{'numerator':0,'denominator':1})
        better={r:[2,1,1,4] for r in regions}
        self.assertEqual(select(zeros,better,regions)[0],'R1')
        with self.assertRaises(ValueError):select(zeros,{regions[0]:[1,0,0,0]},regions)

    def test_recovery_pointer_metadata_only(self):
        pointer=self.fixture('resume_pointer.json')
        self.assertTrue(recovery_pointer(pointer,'fixture-config','fixture-state-hash'))
        for key,value in [('file','../state.pt'),('file','C:/state.pt'),('config_sha256','other'),('step',20001),('sha256','changed')]:
            changed={**pointer,key:value}
            with self.assertRaises(ValueError):recovery_pointer(changed,'fixture-config','fixture-state-hash')

    def test_transport_identity_and_cross_seed(self):
        for recipe in ['r0','r1']:
            m=importlib.import_module('terraingain_landslide.runtime.'+recipe+'.launcher.transport')
            f=self.fixture('index.json');f['identity']=m.identity()
            m.validate_index(f,'dominicamaria')
            wrong=copy.deepcopy(f);wrong['identity']['protocol_sha256']='different'
            with self.assertRaises(RuntimeError):m.validate_index(wrong,'dominicamaria')
            wrong=copy.deepcopy(f);wrong['files']['dominicamaria/seed29/pretrained/config.json']={'kind':'metadata','sha256':'fixture'}
            with self.assertRaises(RuntimeError):m.validate_index(wrong,'dominicamaria')
            with self.assertRaises(RuntimeError):m.safe(ROOT,'../escape.json')

    def test_help_and_inactive_train_without_model_import(self):
        code='''import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','torchvision','numpy','matplotlib'}:
   raise RuntimeError('Heavy import during command inspection: '+fullname)
sys.meta_path.insert(0,Block())
sys.path.insert(0,sys.argv.pop(1))
from terraingain_landslide.cli import main
raise SystemExit(main(sys.argv[1:]))
'''
        for args in [['--help'],['train','--help'],['evaluate','--help'],['gate','--help'],['display','--help'],['train','--recipe','r1','--paths',str(ROOT/'configs/paths.example.yaml'),'--shard','italy','--seed','29']]:
            r=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'src'),*args],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)

    def test_notebook_outputs_and_default_gate(self):
        for file in (ROOT/'notebooks').glob('*.ipynb'):
            d=json.loads(file.read_text())
            for c in d['cells']:
                if c['cell_type']=='code':
                    self.assertIsNone(c['execution_count']);self.assertEqual(c['outputs'],[])
                    ast.parse(''.join(c['source']))
            if file.name=='train_pair.ipynb':
                text='\n'.join(''.join(c['source']) for c in d['cells'])
                self.assertIn('START_TRAINING = False',text)
                self.assertIn('if START_TRAINING:',text)

if __name__=='__main__':unittest.main()

