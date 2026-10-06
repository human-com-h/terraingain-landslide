"""Use current torch AMP API, with explicit PyTorch 2.2 compatibility for local tests."""
import torch

def amp_scaler(**kwargs):
 if hasattr(torch.amp,'GradScaler'):return torch.amp.GradScaler('cuda',**kwargs)
 return torch.cuda.amp.GradScaler(**kwargs)

def amp_autocast():return torch.autocast(device_type='cuda',dtype=torch.float16)
