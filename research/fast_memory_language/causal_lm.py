"""Attention-free causal NCA byte language model, with exact incremental caching.

This is a tested architecture/training starter, NOT a natural-corpus pretrained model.
Each byte position is a cell; the same local rule is reused across iterations.
The associative-memory experiment is separate: its periodic query protocol must
not be applied to a teacher-forced token grid, where it would leak future tokens.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import torch
from torch import Tensor,nn
import torch.nn.functional as F

@dataclass(frozen=True)
class LMConfig:
    vocab_size:int=256
    channels:int=32
    hidden:int=64
    radius:int=2
    iterations:int=8
    def __post_init__(self):
        if min(self.vocab_size,self.channels,self.hidden,self.radius,self.iterations)<1:
            raise ValueError('All configuration dimensions must be positive')

class CausalNCALM(nn.Module):
    def __init__(self,config:LMConfig=LMConfig()):
        super().__init__();self.config=config;c=config
        self.embedding=nn.Embedding(c.vocab_size,c.channels)
        self.perception=nn.Conv1d(c.channels,c.hidden,c.radius+1,bias=False)
        self.anchor=nn.Conv1d(c.channels,c.hidden,1)
        self.candidate=nn.Conv1d(c.hidden,c.channels,1)
        self.gate=nn.Conv1d(c.hidden,c.channels,1)
        self.head=nn.Conv1d(c.channels,c.vocab_size,1)
        nn.init.constant_(self.gate.bias,-1.)

    def _update(self,state:Tensor,local:Tensor,anchor:Tensor)->Tensor:
        h=F.silu(local+anchor);g=torch.sigmoid(self.gate(h))
        return (1-g)*state+g*torch.tanh(self.candidate(h))

    def forward(self,tokens:Tensor)->Tensor:
        if tokens.ndim!=2 or tokens.shape[1]<1:raise ValueError('tokens must be nonempty [B,T]')
        if tokens.dtype!=torch.long:raise TypeError('tokens must be torch.long')
        state=torch.tanh(self.embedding(tokens)).transpose(1,2)
        anchor=self.anchor(state)
        for _ in range(self.config.iterations):
            local=self.perception(F.pad(state,(self.config.radius,0)))
            state=self._update(state,local,anchor)
        return self.head(state).transpose(1,2)

    def initial_cache(self,batch:int)->list[Tensor]:
        if batch<1:raise ValueError('batch must be positive')
        return [self.embedding.weight.new_zeros(batch,self.config.channels,0)
                for _ in range(self.config.iterations)]

    @torch.no_grad()
    def stream_step(self,token:Tensor,cache:list[Tensor]|None=None):
        """Process one byte; return logits for the next byte and per-stage cache.

        Cache k contains predecessor cells' states BEFORE iteration k. Keeping
        only final states and reusing them at every iteration is NOT equivalent.
        Cache is per stream and must be reset at an independent document boundary.
        """
        if token.ndim!=1 or token.dtype!=torch.long:raise ValueError('token must be long [B]')
        if cache is None:cache=self.initial_cache(len(token))
        c=self.config
        if len(cache)!=c.iterations:raise ValueError('Wrong number of cached stages')
        state=torch.tanh(self.embedding(token)).unsqueeze(-1);anchor=self.anchor(state)
        new=[]
        for previous in cache:
            if previous.ndim!=3 or previous.shape[:2]!=state.shape[:2] or previous.shape[-1]>c.radius:
                raise ValueError('Invalid cached state dimensions')
            if previous.device!=state.device or previous.dtype!=state.dtype:
                raise ValueError('Cache device/dtype mismatch')
            window=torch.cat([previous,state],-1)
            new.append(window[:,:,-c.radius:].clone())
            window=F.pad(window,(c.radius+1-window.shape[-1],0))
            state=self._update(state,self.perception(window),anchor)
        return self.head(state).squeeze(-1),new

    def max_context_bytes(self)->int:
        return 1+self.config.radius*self.config.iterations
    def parameter_count(self)->int:return sum(p.numel() for p in self.parameters())
    def config_dict(self):return asdict(self.config)

@torch.no_grad()
def generate(model:CausalNCALM,prompt:bytes,count:int=100,temperature:float=1.,seed:int=0)->bytes:
    if not prompt:raise ValueError('Prompt must have at least one byte')
    if count<0 or temperature<=0:raise ValueError('Invalid generation options')
    device=model.embedding.weight.device
    g=torch.Generator(device=device).manual_seed(seed);cache=None;out=bytearray(prompt)
    for b in prompt:logits,cache=model.stream_step(torch.tensor([b],device=device),cache)
    for _ in range(count):
        token=torch.multinomial(torch.softmax(logits/temperature,-1),1,generator=g).flatten()
        out.append(int(token.item()));logits,cache=model.stream_step(token,cache)
    return bytes(out)
