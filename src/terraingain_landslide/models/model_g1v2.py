"""SatMAE dense adaptation; source provenance and CC BY-NC 4.0 in accompanying files."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
BANDS=['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
GROUPS=((0,1,2,6),(3,4,5,7),(8,9))
class PatchEmbed(nn.Module):
 def __init__(self,c):
  super().__init__();self.proj=nn.Conv2d(c,768,8,8)
 def forward(self,x):return self.proj(x).flatten(2).transpose(1,2)
class Attention(nn.Module):
 def __init__(self):
  super().__init__();self.qkv=nn.Linear(768,2304);self.proj=nn.Linear(768,768);self.reference=False
 def forward(self,x):
  b,n,c=x.shape;q,k,v=self.qkv(x).reshape(b,n,3,12,64).permute(2,0,3,1,4).unbind(0)
  y=((q*64**-.5)@k.transpose(-2,-1)).softmax(-1)@v if self.reference else F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
  return self.proj(y.transpose(1,2).reshape(b,n,c))
class MLP(nn.Module):
 def __init__(self):
  super().__init__();self.fc1=nn.Linear(768,3072);self.fc2=nn.Linear(3072,768);self.act=nn.GELU()
 def forward(self,x):return self.fc2(self.act(self.fc1(x)))
class Block(nn.Module):
 def __init__(self):
  super().__init__();self.norm1=nn.LayerNorm(768,eps=1e-6);self.attn=Attention();self.norm2=nn.LayerNorm(768,eps=1e-6);self.mlp=MLP()
 def forward(self,x):
  x=x+self.attn(self.norm1(x));return x+self.mlp(self.norm2(x))
class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.patch_embed=nn.ModuleList([PatchEmbed(len(g)) for g in GROUPS])
  self.cls_token=nn.Parameter(torch.zeros(1,1,768));self.pos_embed=nn.Parameter(torch.zeros(1,257,512))
  self.channel_embed=nn.Parameter(torch.zeros(1,3,256));self.channel_cls_embed=nn.Parameter(torch.zeros(1,1,256))
  self.blocks=nn.ModuleList([Block() for _ in range(12)]);self.norm=nn.LayerNorm(768,eps=1e-6)
 def forward(self,x):
  if x.shape[1:]!=(10,128,128):raise ValueError('Require unchanged 10-band 128x128 input')
  z=torch.stack([p(x[:,g]) for p,g in zip(self.patch_embed,GROUPS)],1)
  pos=self.pos_embed[:,1:].unsqueeze(1).expand(-1,3,-1,-1)
  ch=self.channel_embed.unsqueeze(2).expand(-1,-1,256,-1)
  z=(z+torch.cat((pos,ch),-1)).reshape(x.shape[0],768,768)
  cls=self.cls_token+torch.cat((self.pos_embed[:,:1],self.channel_cls_embed),-1)
  z=torch.cat((cls.expand(x.shape[0],-1,-1),z),1)
  for block in self.blocks:z=block(z)
  z=self.norm(z)[:,1:].reshape(x.shape[0],3,16,16,768)
  return z.permute(0,1,4,2,3).reshape(x.shape[0],2304,16,16)
def conv(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(8,b),nn.GELU())
class Segmenter(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=Encoder()
  self.decoder=nn.ModuleDict({'project':nn.Sequential(nn.Conv2d(2304,128,1,bias=False),nn.GroupNorm(8,128),nn.GELU()),'up1':conv(128,128),'up2':conv(128,64),'up3':conv(64,32),'head':nn.Conv2d(32,1,1)})
 def forward(self,x):
  z=self.decoder['project'](self.encoder(x))
  for k in ['up1','up2','up3']:z=self.decoder[k](F.interpolate(z,scale_factor=2,mode='bilinear',align_corners=False))
  return self.decoder['head'](z)[:,0]
def interpolate_position(p):
 assert p.shape==(1,145,512),tuple(p.shape)
 grid=p[:,1:].reshape(1,12,12,512).permute(0,3,1,2)
 grid=F.interpolate(grid,size=(16,16),mode='bicubic',align_corners=False).permute(0,2,3,1).reshape(1,256,512)
 return torch.cat((p[:,:1],grid),1)
def load_encoder(model,path):
 state=torch.load(path,map_location='cpu',weights_only=True)
 expect=model.encoder.state_dict();assert set(state)==set(expect)-{'channel_cls_embed'}
 for i,c in enumerate([4,4,2]):assert state[f'patch_embed.{i}.proj.weight'].shape==(768,c,8,8)
 state['pos_embed']=interpolate_position(state['pos_embed'])
 state['channel_cls_embed']=torch.zeros_like(expect['channel_cls_embed'])
 model.encoder.load_state_dict(state,strict=True);return model
def full_batch_logits(model,x,microbatch=2,checkpointing=True):
 # Checkpoint each microbatch but keep a single, exact batch-8 Dice loss.
 return torch.cat([checkpoint(model,a,use_reentrant=False,preserve_rng_state=True) if checkpointing and model.training else model(a) for a in x.split(microbatch)],0)
def loss_fn(logits,truth,positive_weight):
 p=logits.float().sigmoid();y=truth.float()
 return F.binary_cross_entropy_with_logits(logits.float(),y,pos_weight=positive_weight.float())+1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1)

