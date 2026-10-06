"""Execution-only wrapper: one frozen P/S pair per Kaggle session; no scientific overrides."""
import argparse, subprocess, sys, os, time, signal, shutil, contextlib, traceback
from pathlib import Path
from .transport import *

@contextlib.contextmanager
def lock(work):
    p=Path(work)/'RUNNING.lock'
    try:fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('This shard workspace is already claimed. Do not launch twice.')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:yield
    finally:p.unlink(missing_ok=True)

def prepare(work,shard,inputs,execution_seed=None):
    verify_delivery();sys.path.insert(0,str(CORE));from ..core.common_g1v2 import verify_data
    verify_data(shard)
    for source in inputs:
        for f in Path(source).rglob('*'):
            if not f.is_file():continue
            require(f.name!=shard+'.npz','Outer source array mounted during training')
            if f.suffix=='.zip':
                with zipfile.ZipFile(f) as z:require(not any(Path(n).name==shard+'.npz' for n in z.namelist()),'Outer source archive mounted during training')
    idx=restore(work,shard,inputs,execution_seed)
    require(not list((Path(work)/'results').rglob('failure_*.json')),'Prior technical failure: preserve output; no automatic retry')
    complete_runs(Path(work)/'results',shard,idx['files'],execution_seed)
    return idx

def disk_budget(work,free_bytes=None):
    import math
    # Model float32 + AdamW first/second moments, plus RNG/scaler/history/container allowance.
    elements=sum(math.prod(shape) for shape in read(CORE/'MODEL_STRUCTURE.json').values())
    state_bytes=elements*12+16*1024**2
    # One pair: max 8 retained full states at normal pause/completion, plus 8 ZIP_STORED copies.
    # During rolling replacement/20k commit transient state count is below this 16-copy peak.
    metadata_and_margin=1024**3
    planned_peak=16*state_bytes+metadata_and_margin
    work=Path(work);existing=0
    # Credit only active-pair states that are part of the planned peak, never logs/junk/old sessions.
    if (work/'SESSION.json').exists():
        seed=read(work/'SESSION.json')['seed']
        for run in (work/'results').glob(f'*/seed{seed}/*'):
            states=list(run.glob('checkpoint_*/state.pt'))
            if (run/'resume_pointer.json').exists():states.append(safe(run,read(run/'resume_pointer.json')['file']))
            existing+=sum(f.stat().st_size for f in states if f.is_file())
    existing=min(existing,8*state_bytes)
    free=shutil.disk_usage(work).free if free_bytes is None else free_bytes
    required=max(2*1024**3,planned_peak-existing)
    return dict(model_elements=elements,estimated_full_state_bytes=state_bytes,retained_states_per_pair=8,
                state_and_export_copies=16,metadata_and_margin_bytes=metadata_and_margin,
                planned_peak_bytes=planned_peak,existing_work_bytes=existing,free_bytes=free,
                required_free_bytes=required,status='PASS' if free>=required else 'FAIL')

def preflight(work,shard,deadline,execution_seed=None):
    import torch, torchvision
    p=read(CORE/'S05F_PROTOCOL.json')['recipe'];root=Path(work)/'results'/shard
    if execution_seed is not None:root=root/f'seed{execution_seed}'
    tag=str(time.time_ns())
    env=dict(torch=torch.__version__,torchvision=torchvision.__version__,cuda=torch.version.cuda,devices=[],free_disk_bytes=shutil.disk_usage(work).free)
    require(torch.cuda.is_available() and torch.cuda.device_count()==2,'Exactly two CUDA GPUs required')
    for i in range(2):
        free,total=torch.cuda.mem_get_info(i);name=torch.cuda.get_device_name(i)
        env['devices'].append(dict(index=i,name=name,free_bytes=free,total_bytes=total))
        require(name=='Tesla T4' and free>=12*1024**3,'T4 with at least 12 GiB free required')
    require(torch.__version__==p['environment']['kaggle_torch'] and torchvision.__version__==p['environment']['kaggle_torchvision'],'Frozen PyTorch/torchvision mismatch; stop, no package installation or protocol changes')
    budget=disk_budget(work);env['disk_budget']=budget
    write(root/f'PREFLIGHT_DISK_{tag}.json',budget)
    print(f"DISK: free={budget['free_bytes']/1024**3:.2f} GiB; required={budget['required_free_bytes']/1024**3:.2f} GiB; estimated pair+export peak={budget['planned_peak_bytes']/1024**3:.2f} GiB",flush=True)
    require(budget['status']=='PASS',f"Insufficient workspace/export disk: free={budget['free_bytes']} bytes, required={budget['required_free_bytes']} bytes; inspect PREFLIGHT_DISK report")
    write(root/f'PREFLIGHT_ENVIRONMENT_{tag}.json',env)
    print('Starting synthetic preflight on BOTH T4s; no formal training before both pass',flush=True)
    procs=[];handles=[]
    try:
        for gpu,arm in enumerate(ARMS):
            log=(root/f'preflight_{arm}_{tag}.log').open('w',encoding='utf-8');handles.append(log)
            envp=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
            procs.append(subprocess.Popen([sys.executable,'-m','terraingain_landslide.runtime.r0.core.engineering_tests','--preflight','--output',str(root/f'preflight_{arm}_{tag}.json')],env=envp,stdout=log,stderr=subprocess.STDOUT))
        codes=[p.wait(timeout=max(1,deadline-time.time())) for p in procs]
        require(codes==[0,0],'Preflight failed; inspect both GPU reports/logs; no training launched')
        for arm in ARMS:
            r=read(root/f'preflight_{arm}_{tag}.json');require(r['status']=='PASS' and r['synthetic_only'] and r['scientific_training_increment']==0,'Invalid preflight report')
    finally:
        for p in procs:
            if p.poll() is None:p.terminate();p.wait()
        for h in handles:h.close()
    require(time.time()<deadline,'Session budget used by preflight; no training launched')
    print('BOTH T4 PREFLIGHTS PASS: checkpoint, prediction and full-state resume tested',flush=True)

def train_shard(work,shard,inputs,destination,hours,authorized,execution_seed=None):
    require(authorized,'Explicit --authorize-training is required on Kaggle')
    require(0<hours<=6,'Session hours must be in (0,6]')
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    with lock(work):
        idx=prepare(work,shard,inputs,execution_seed);seed=selected_seed(idx)
        if seed is None:print(f'{shard}: {len(idx["complete"])}/{len(runs(shard,execution_seed))} complete. No training.');return
        session=work/'SESSION.json'
        if session.exists():
            state=read(session);require(state['seed']==seed,'One seed pair per session. Save Output and start a fresh session for the next seed.')
            require(time.time()<state['deadline'],'This session budget expired. Save Output and start a fresh session.')
        else:
            # 30 minutes reserved inside the <=6 h session for state flush and verified export.
            require(hours>0.5,'Session must allow the 30-minute export reserve')
            state=dict(shard=shard,seed=seed,started=time.time(),deadline=time.time()+(hours*3600-1800));write(session,state)
        root=work/'results';session_root=root/shard
        if execution_seed is not None:session_root=session_root/f'seed{execution_seed}'
        session_root.mkdir(parents=True,exist_ok=True)
        write(session_root/f'execution_{time.time_ns()}.json',dict(**state,identity=identity(),scientific_count_before=607,automatic_followon=False))
        children=[];logs=[];handlers={}
        def stop(signum,frame):
            for child in children:
                if child.poll() is None:child.send_signal(signal.SIGTERM)
        try:
            materialize(work,idx,seed,inputs)
            preflight(work,shard,state['deadline'],execution_seed)
            authorization=work/'AUTHORIZATION.json'
            write(authorization,dict(explicit_user_authorization=True,protocol_sha256=identity()['protocol_sha256'],allowed_pairs=[[shard,s] for s in (SEEDS if execution_seed is None else (execution_seed,))],max_formal_runs=30,scientific_count_before=607))
            shutil.copyfile(authorization,session_root/'AUTHORIZATION.json')
            for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,stop)
            for gpu,arm in enumerate(ARMS):
                run=f'{shard}/seed{seed}/{arm}'
                if run in idx['complete']:print('SKIP COMPLETE',run,flush=True);continue
                logdir=root/run;logdir.mkdir(parents=True,exist_ok=True)
                log=(logdir/f'worker_{time.time_ns()}.log').open('w',encoding='utf-8');logs.append(log)
                cmd=[sys.executable,'-m','terraingain_landslide.runtime.r0.core.train_s05f','--output',str(root),'--split',shard,'--seed',str(seed),'--arm',arm,'--authorization',str(authorization),'--deadline',str(state['deadline'])]
                children.append(subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT))
            heartbeat=0
            while any(c.poll() is None for c in children):
                if time.time()-heartbeat>=60:
                    progress={}
                    for arm in ARMS:
                        pointer=root/shard/f'seed{seed}'/arm/'resume_pointer.json'
                        curve=root/shard/f'seed{seed}'/arm/'curve.json'
                        progress[arm]=read(pointer)['step'] if pointer.exists() else (read(curve)[-1]['attempt'] if curve.exists() else 0)
                    print('SAVED STEPS',shard,seed,progress,flush=True);heartbeat=time.time()
                if any(c.poll() not in (None,0,75) for c in children):stop(None,None)
                time.sleep(2)
            require(all(c.returncode in (0,75) for c in children),'Technical training failure; no automatic retry')
            write(session_root/'SESSION_STATUS.json',dict(seed=seed,exit_codes=[c.returncode for c in children],status='COMPLETE_PAIR' if all(c.returncode==0 for c in children) else 'PAUSED',automatic_followon=False))
        except BaseException as exc:
            stop(None,None)
            for c in children:
                if c.poll() is None:c.wait()
            write(session_root/f'failure_launcher_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),automatic_followon=False));raise
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
            for f in logs:f.close()
            # Core failure records are written at the results root; keep them within this shard's transport.
            for f in root.glob('failure_*.json'):f.replace(session_root/f.name)
            export(work,shard,destination,inputs)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train','export']);p.add_argument('--shard',choices=['dominicamaria','italy','hiroshima','hokkaido','thrissur']);p.add_argument('--work',type=Path,required=True);p.add_argument('--input',type=Path,action='append',default=[]);p.add_argument('--export-dir',type=Path);p.add_argument('--session-hours',type=float,default=6);p.add_argument('--authorize-training',action='store_true');p.add_argument('--seed',type=int,choices=SEEDS,default=None);a=p.parse_args()
    require(a.shard is not None and a.export_dir is not None,'Shard and export directory required')
    if a.mode=='train':train_shard(a.work,a.shard,a.input,a.export_dir,a.session_hours,a.authorize_training,a.seed)
    elif a.mode=='prepare':
        idx=prepare(a.work,a.shard,a.input,a.seed);print('SHARD',a.shard,'SCOPE SEED',a.seed,'COMPLETE',len(idx['complete']),'/'+str(len(runs(a.shard,a.seed))),'NEXT SEED',selected_seed(idx))
    else:
        a.work.mkdir(parents=True,exist_ok=True)
        with lock(a.work):
            verify_delivery();restore(a.work,a.shard,a.input,a.seed);export(a.work,a.shard,a.export_dir,a.input)
if __name__=='__main__':main()
