"""Offline shard transport. Frozen core is never modified. Metadata index + immutable state ZIPs."""
from pathlib import Path, PurePosixPath
import hashlib, json, zipfile, shutil, os, io, builtins, contextlib
CORE = Path(__file__).resolve().parents[1] / 'core'
SEEDS=(17,29,43)
ARMS=('pretrained','scratch')

def require(ok, message):
    if not ok: raise RuntimeError(message)
def sha(path):
    with Path(path).open('rb') as f: return digest(f)
def digest(f):
    h=hashlib.sha256()
    for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(p)
def safe(root,rel):
    q=PurePosixPath(rel)
    require(not q.is_absolute() and '..' not in q.parts and '\\' not in rel and ':' not in rel,'Unsafe member '+rel)
    p=Path(root).joinpath(*q.parts)
    require(p.resolve().is_relative_to(Path(root).resolve()),'Escaping member');return p

def identity():
    return dict(protocol_sha256=sha(CORE/'S05F_PROTOCOL.json'),core_manifest_sha256=sha(CORE/'PACKAGE_MANIFEST.json'))
def verify_delivery():
    from terraingain_landslide.utils.integrity import verify_source_tree
    verify_source_tree()
    base=CORE.parent; m=read(base/'DELIVERY_MANIFEST.json')
    for rel,h in m['files'].items(): require(sha(safe(base,rel))==h,'Delivery hash mismatch: '+rel)
    for rel,h in read(CORE/'PACKAGE_MANIFEST.json')['files'].items():require(sha(safe(CORE,rel))==h,'Frozen core changed: '+rel)
    return identity()
def runs(shard,seed=None): return [f'{shard}/seed{s}/{a}' for s in (SEEDS if seed is None else (seed,)) for a in ARMS]

def resume_name(shard,seed=None):return f'S05F_{shard}_resume.zip' if seed is None else f'S05F_{shard}_seed{seed}_resume.zip'
def validate_index(idx,shard):
    require(shard in ('dominicamaria','italy','hiroshima','hokkaido','thrissur') and idx['shard']==shard,'Cross-shard input rejected')
    require(idx['identity']==identity(),'Protocol/core mismatch')
    require(idx['schema']==1,'Unsupported index')
    seed=idx.get('execution_seed');require(seed is None or seed in SEEDS,'Invalid execution seed')
    for rel,r in idx['files'].items():
        safe(Path.cwd(),rel)
        require(rel.split('/')[0]==shard,'Cross-shard member')
        if seed is not None:require(rel.startswith(f'{shard}/seed{seed}/'),'Cross-seed member')
        require(r['kind'] in ('metadata','blob'),'Bad storage')
        if r['kind']=='blob':
            require(r['blob']==f'S05F_{shard}_blob_{r["sha256"]}.zip','Blob name mismatch')
            require(rel.endswith('.pt'),'Unexpected blob type')
    require(set(idx['complete']).issubset(runs(shard,seed)),'Unexpected run')

class DirectoryArchive:
    """Read-only archive interface for a Dataset-expanded resume index."""
    def __init__(self,path):self.root=Path(path).parent
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def namelist(self):return ['INDEX.json']+['results/'+p.relative_to(self.root/'results').as_posix() for p in (self.root/'results').rglob('*') if p.is_file()]
    def open(self,name):return safe(self.root,name).open('rb')
    def read(self,name):return safe(self.root,name).read_bytes()

def index_archive(path):return DirectoryArchive(path) if Path(path).name=='INDEX.json' else zipfile.ZipFile(path)

def load_index(path,shard):
    with index_archive(path) as z:
        require(len(z.namelist())==len(set(z.namelist())),'Duplicate index members')
        idx=json.loads(z.read('INDEX.json'));validate_index(idx,shard)
        expected={'INDEX.json'}|{'results/'+r for r,v in idx['files'].items() if v['kind']=='metadata'}
        require(set(z.namelist())==expected,'Unlisted index metadata members')
        for rel,v in idx['files'].items():
            if v['kind']=='metadata':
                with z.open('results/'+rel) as f:require(digest(f)==v['sha256'],'Metadata hash mismatch')
    return idx

def extracted_head(path,shard,seed,inputs):
    stem=resume_name(shard,seed).removesuffix('.zip');wanted=sha(path);origins=[]
    for root in inputs:
        for p in Path(root).rglob(stem+'_INDEX_ORIGIN.json'):
            r=read(p)
            if r['index_sha256']==wanted and r['archive_name']==resume_name(shard,seed):origins.append(r['archive_sha256'])
    require(len(set(origins))==1,'Extracted resume needs matching '+stem+'_INDEX_ORIGIN.json; keep the whole export or original resume ZIP')
    return origins[0]

def latest(inputs,shard,seed=None):
    candidates={};raw=[]
    for root in inputs:
        for p in Path(root).rglob(resume_name(shard,seed)):
            h=sha(p)
            if h not in candidates:
                idx=load_index(p,shard);require(idx.get('execution_seed')==seed,'Index scope mismatch');candidates[h]=(p,idx)
        for p in Path(root).rglob('INDEX.json'):
            idx=read(p)
            if idx.get('shard')==shard and idx.get('execution_seed')==seed:raw.append(p)
    for p in raw:
        idx=load_index(p,shard);h=extracted_head(p,shard,seed,inputs)
        if h not in candidates:candidates[h]=(p,idx)
    if not candidates:return None
    h,(p,idx)=max(candidates.items(),key=lambda item:item[1][1]['generation'])
    require(all(k==h or k in idx['ancestors'] for k in candidates),'Forked/duplicate execution histories; do not choose silently')
    return p,idx,h

def resolve_blob(record,inputs):
    found=[];raw=[]
    for root in inputs:
        found.extend(Path(root).rglob(record['blob']))
        raw.extend(Path(root).rglob(record['blob'].removesuffix('.zip')+'.pt'))
    require(bool(found or raw),'Missing historical state: '+record['blob'])
    for p in raw:require(p.stat().st_size==record['size'] and sha(p)==record['sha256'],'Raw state size/hash mismatch')
    for p in found:
        with zipfile.ZipFile(p) as z:
            require(z.namelist()==['state.pt'],'Unexpected blob members')
            require(z.getinfo('state.pt').file_size==record['size'],'Blob size mismatch')
            with z.open('state.pt') as f:require(digest(f)==record['sha256'],'Blob hash mismatch')
    return raw[0] if raw else found[0]

def complete_runs(root,shard,files,seed=None):
    expected=set(runs(shard,seed));done=[]
    actual={p.parent.relative_to(root).as_posix() for p in (Path(root)/shard).rglob('result.json')}
    require(actual.issubset(expected),'Unexpected/duplicate run path')
    for run in sorted(actual):
        result=read(Path(root)/run/'result.json');cfg=read(Path(root)/run/'config.json')
        require(result['status']=='COMPLETE' and result['primary_step']==20000 and result['checkpoint_set']==[5000,10000,15000,20000],'Invalid completion receipt')
        require(cfg['protocol_sha256']==identity()['protocol_sha256'] and cfg['package_sha256']==identity()['core_manifest_sha256'],'Run identity mismatch')
        require(run==f'{cfg["split"]}/seed{cfg["seed"]}/{cfg["arm"]}','Run/config mismatch')
        needed={'config.json','curve.json','INITIALIZATION.json','SELECTION.json'}|{f'checkpoint_{s:05d}/{n}' for s in (5000,10000,15000,20000) for n in ('state.pt','train.npz','validation.npz','checkpoint.json','MANIFEST.json')}
        require(needed.issubset(result['files']),'Incomplete completion inventory')
        for rel,h in result['files'].items():
            name=run+'/'+rel;safe(root,name)
            require(name in files and files[name]['sha256']==h,'Receipt mismatch: '+name)
            p=Path(root)/name
            if files[name]['kind']=='metadata':require(p.exists() and sha(p)==h,'Missing/changed metadata: '+name)
        done.append(run)
    return done

def restore(work,shard,inputs,seed=None):
    work=Path(work);work.mkdir(parents=True,exist_ok=True);root=work/'results';root.mkdir(exist_ok=True)
    ledger=work/'LEDGER.json'
    if ledger.exists():
        idx=read(ledger);validate_index(idx,shard);require(idx.get('execution_seed')==seed,'Workspace seed scope differs');return idx
    require(not any(root.iterdir()),'Nonempty workspace without ledger')
    source=latest(inputs,shard,seed)
    if source:
        p,idx,h=source
        with index_archive(p) as z:
            for rel,v in idx['files'].items():
                if v['kind']=='metadata':
                    out=safe(root,rel);out.parent.mkdir(parents=True,exist_ok=True)
                    with z.open('results/'+rel) as src,out.open('wb') as dst:shutil.copyfileobj(src,dst)
        idx=dict(idx,head_sha256=h)
        require(complete_runs(root,shard,idx['files'],seed)==idx['complete'],'Completion inventory mismatch')
    else:idx=dict(schema=1,shard=shard,identity=identity(),generation=0,ancestors=[],head_sha256=None,files={},complete=[])
    if seed is not None:idx['execution_seed']=seed
    write(ledger,idx);return idx

def selected_seed(idx):
    return next((s for s in (SEEDS if idx.get('execution_seed') is None else (idx['execution_seed'],)) if any(f'{idx["shard"]}/seed{s}/{a}' not in idx['complete'] for a in ARMS)),None)

def materialize(work,idx,seed,inputs):
    root=Path(work)/'results';prefix=f'{idx["shard"]}/seed{seed}/'
    for rel,r in idx['files'].items():
        if r['kind']!='blob' or not rel.startswith(prefix):continue
        # Completed arm need not be loaded by the frozen trainer.
        if '/'.join(rel.split('/')[:3]) in idx['complete']:continue
        out=safe(root,rel)
        if out.exists():require(sha(out)==r['sha256'],'Local state changed');continue
        blob=resolve_blob(r,inputs);out.parent.mkdir(parents=True,exist_ok=True)
        if blob.suffix=='.pt':shutil.copyfile(blob,out)
        else:
            with zipfile.ZipFile(blob) as z,z.open('state.pt') as src,out.open('wb') as dst:shutil.copyfileobj(src,dst)
        require(sha(out)==r['sha256'],'Restored state mismatch')

def export(work,shard,destination,inputs=()):
    work=Path(work);root=work/'results';dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    idx=read(work/'LEDGER.json');validate_index(idx,shard);seed=idx.get('execution_seed')
    files={rel:r for rel,r in idx['files'].items() if r['kind']=='blob'}
    for rel in list(files):
        p=root/rel
        if p.name.startswith('resume_'):
            pointer=p.parent/'resume_pointer.json'
            if not pointer.exists() or read(pointer)['file']!=p.name:del files[rel]
    for p in sorted(root.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(root).as_posix()
        if any(part.endswith('.tmp') for part in p.relative_to(root).parts):continue
        require(rel.split('/')[0]==shard,'Cross-shard export')
        if p.name.startswith('resume_') and p.suffix=='.pt':
            pointer=p.parent/'resume_pointer.json'
            if not pointer.exists() or read(pointer)['file']!=p.name:continue
        h=sha(p);r=dict(sha256=h,size=p.stat().st_size,kind='blob' if p.suffix=='.pt' else 'metadata')
        if r['kind']=='blob':
            r['blob']=f'S05F_{shard}_blob_{h}.zip';target=dest/(r['blob'] if seed is None else r['blob'].removesuffix('.zip')+'.pt')
            if target.exists():resolve_blob(r,[dest])
            else:
                require(shutil.disk_usage(dest).free>r['size']+256*1024**2,'Insufficient export disk')
                tmp=target.with_name(target.name+'.tmp')
                if seed is None:
                    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_STORED,allowZip64=True) as z:z.write(p,'state.pt')
                else:shutil.copyfile(p,tmp)
                tmp.replace(target);resolve_blob(r,[dest])
        files[rel]=r
    # Preserve the already-completed arm when the other arm is resumed.
    active=read(work/'SESSION.json')['seed'] if (work/'SESSION.json').exists() else selected_seed(idx)
    for rel,r in files.items():
        name=r.get('blob','');target=dest/(name if seed is None else name.removesuffix('.zip')+'.pt')
        if r['kind']=='blob' and active and rel.startswith(f'{shard}/seed{active}/') and not target.exists():
            src=resolve_blob(r,inputs)
            if seed is not None and src.suffix=='.zip':
                with zipfile.ZipFile(src) as z,z.open('state.pt') as f,target.open('wb') as g:shutil.copyfileobj(f,g)
            elif seed is None and src.suffix=='.pt':
                with zipfile.ZipFile(target,'w',zipfile.ZIP_STORED,allowZip64=True) as z:z.write(src,'state.pt')
            else:shutil.copyfile(src,target)
            resolve_blob(r,[dest])
    done=complete_runs(root,shard,files,seed)
    ancestors=idx['ancestors']+([idx['head_sha256']] if idx.get('head_sha256') else [])
    new=dict(schema=1,shard=shard,identity=identity(),generation=idx['generation']+1,ancestors=ancestors,files=files,complete=done,scientific_count_before=607,automatic_followon=False)
    if seed is not None:new['execution_seed']=seed
    target=dest/resume_name(shard,seed);tmp=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for rel,r in files.items():
            if r['kind']=='metadata':z.write(root/rel,'results/'+rel)
        z.writestr('INDEX.json',json.dumps(new,indent=2))
    load_index(tmp,shard);tmp.replace(target);new['head_sha256']=sha(target);write(work/'LEDGER.json',new)
    with zipfile.ZipFile(target) as z:index_hash=hashlib.sha256(z.read('INDEX.json')).hexdigest()
    write(dest/(target.stem+'_INDEX_ORIGIN.json'),dict(archive_name=target.name,archive_sha256=new['head_sha256'],index_sha256=index_hash))
    write(dest/(f'S05F_{shard}_SHA256.json' if seed is None else f'S05F_{shard}_seed{seed}_SHA256.json'),{p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file() and p.suffix in ('.zip','.pt')})
    print(f'EXPORTED {target}; completed {len(done)}/{len(runs(shard,seed))}; retain ALL accompanying state files and INDEX_ORIGIN',flush=True)
    return target

class StoredStateView(io.RawIOBase):
    """Bounded direct file view: O(1) seeks inside ZIP_STORED (no ZipExtFile replay)."""
    def __init__(self, archive):
        super().__init__()
        import struct
        with zipfile.ZipFile(archive) as z:
            info=z.getinfo('state.pt')
            require(info.compress_type==zipfile.ZIP_STORED and not info.flag_bits & 1,'State blob must be unencrypted ZIP_STORED')
            offset=info.header_offset;self.length=info.file_size
        self.raw=io.open(archive,'rb');self.raw.seek(offset);header=self.raw.read(30)
        require(len(header)==30 and header[:4]==b'PK\x03\x04','Invalid local ZIP header')
        fields=struct.unpack('<IHHHHHIIIHH',header)
        self.start=offset+30+fields[9]+fields[10];self.position=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        require(whence in (0,1,2),'Invalid seek mode')
        position=offset+(self.position if whence==1 else self.length if whence==2 else 0)
        if position<0:raise ValueError('Negative seek')
        self.position=position;return position
    def readinto(self,buffer):
        count=min(len(buffer),max(0,self.length-self.position))
        self.raw.seek(self.start+self.position);data=self.raw.read(count)
        buffer[:len(data)]=data;self.position+=len(data);return len(data)
    def close(self):
        if hasattr(self,'raw'):self.raw.close()
        super().close()

@contextlib.contextmanager
def virtual_states(mapping):
    """Read-only seekable ZIP_STORED streams for the unchanged assessor's Path/torch reads."""
    original_open=builtins.open;original_io=io.open
    opened=[]
    def intercept(original):
        def wrapped(file,mode='r',*args,**kwargs):
            key=str(Path(file).resolve()) if isinstance(file,(str,bytes,os.PathLike)) and not isinstance(file,bytes) else None
            if key in mapping:
                require(mode=='rb','Virtual state supports read-only binary access')
                source=Path(mapping[key])
                f=original_io(source,'rb') if source.suffix=='.pt' else io.BufferedReader(StoredStateView(source),buffer_size=1024*1024)
                opened.append(f);return f
            return original(file,mode,*args,**kwargs)
        return wrapped
    builtins.open=intercept(original_open);io.open=intercept(original_io)
    try:yield
    finally:
        builtins.open=original_open;io.open=original_io
        for f in opened:f.close()

