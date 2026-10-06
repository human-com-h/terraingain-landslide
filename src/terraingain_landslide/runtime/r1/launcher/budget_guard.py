"""Wall-cap process group guard and GPU allocation receipt. No automatic retries."""
import datetime,json,os,signal,subprocess,time
from pathlib import Path

def guarded_run(command,cap_seconds,started_monotonic,allocation_id,output,gpu_count=2):
 if gpu_count!=2 or cap_seconds<=1800:raise ValueError('Invalid dual-T4 allocation')
 elapsed=time.monotonic()-started_monotonic;remaining=cap_seconds-elapsed
 if remaining<=1800:raise RuntimeError('Allocation consumed by preparation; no worker started')
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 record=dict(allocation_id=allocation_id,gpu_count=2,cap_seconds=cap_seconds,
  accounting='Full dual-GPU wall time since command entry; operator allocation/idle outside this interval must be added',
  started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),preparation_seconds=elapsed,
  automatic_retry=False,external_vm_idle_seconds=None)
 process=None
 try:
  process=subprocess.Popen(command,start_new_session=True)
  try:code=process.wait(timeout=max(1,remaining-30))
  except subprocess.TimeoutExpired:
   record['hard_cap_triggered']=True
   os.killpg(process.pid,signal.SIGTERM)
   try:code=process.wait(timeout=max(.1,cap_seconds-(time.monotonic()-started_monotonic)))
   except subprocess.TimeoutExpired:
    os.killpg(process.pid,signal.SIGKILL);code=process.wait()
  record['exit_code']=code
 finally:
  if process is not None and process.poll() is None:
   os.killpg(process.pid,signal.SIGKILL);process.wait()
  record['measured_wall_seconds']=time.monotonic()-started_monotonic
  record['measured_single_T4_equivalent_hours']=2*record['measured_wall_seconds']/3600
  record['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
  record['includes_preflight_restore_export']=True
  (output/(allocation_id+'_BUDGET_RECEIPT.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
 if record.get('exit_code')!=0:raise RuntimeError('Worker stopped or failed. Preserve ALL outputs; no automatic retry. See budget receipt and failure logs.')
 return record
