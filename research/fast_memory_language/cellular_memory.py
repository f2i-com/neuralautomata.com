"""A structured, learned fast-memory cellular model and state-matched controls.

Scope: 1D ring, synchronous nearest-neighbor relay. Routing and the delta
learning law are supplied, NOT discovered. Keys, value encodings and decoding
are learned with an outer optimizer. Test episodes change only private memory.
No global memory lookup, attention, source-value cache or query-target input.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from legacy.fast_memory import write as local_delta_write


@dataclass(frozen=True)
class MemoryConfig:
    symbols: int = 16
    bits: int = 8
    key_dim: int = 16
    value_dim: int = 8
    variant: str = 'fast'
    def __post_init__(self):
        if min(self.symbols,self.bits,self.key_dim,self.value_dim)<1:
            raise ValueError('All widths must be positive')
        if self.variant not in ('fast','frozen_keys','gru'):
            raise ValueError('variant must be fast, frozen_keys or gru')


class CellularMemory(nn.Module):
    """Shared local rule with private adaptive memory and public query packets.

    Private state [B,N,D] has D = value_dim*key_dim in every learned arm.
    Public packet = [key, accumulated value, valid]. A query visits each cell
    once in N synchronous updates and returns to its origin. Zero padding is
    NOT used: the periodic ring is the declared topology.
    """
    def __init__(self, config: MemoryConfig):
        super().__init__(); self.config=config
        c=config; d=c.value_dim*c.key_dim
        self.keys=nn.Embedding(c.symbols,c.key_dim)
        nn.init.normal_(self.keys.weight,std=1.)
        self.keys.weight.requires_grad_(c.variant!='frozen_keys')
        self.value_encoder=nn.Linear(c.bits,c.value_dim,bias=False)
        self.decoder=nn.Linear(c.value_dim,c.bits,bias=False)
        self.log_scale=nn.Parameter(torch.tensor(0.))
        if c.variant=='gru':
            self.writer=nn.GRUCell(c.key_dim+c.value_dim,d)
            self.reader=nn.Sequential(nn.Linear(d+c.key_dim,64),nn.SiLU(),
                                      nn.Linear(64,c.value_dim,bias=False))

    def encoded_keys(self, symbols: Tensor) -> Tensor:
        return F.normalize(self.keys(symbols),dim=-1,eps=1e-8)

    def initial_memory(self, batch: int, cells: int) -> Tensor:
        if batch<1 or cells<2: raise ValueError('Need positive batch and at least two cells')
        return self.keys.weight.new_zeros(batch,cells,self.config.value_dim*self.config.key_dim)

    def write_event(self, memory: Tensor, symbols: Tensor, values: Tensor,
                    positions: Tensor, *, enabled: bool=True) -> Tensor:
        """Write ONLY at supplied local observation sites, one event per episode.

        Values are legitimate observed inputs here, not future query answers.
        Masks are explicit event availability. Thinking updates do not write.
        """
        b,n,d=memory.shape; c=self.config
        if symbols.shape!=(b,) or values.shape!=(b,c.bits) or positions.shape!=(b,):
            raise ValueError('Invalid event shapes')
        if not enabled: return memory
        if not bool(((positions>=0)&(positions<n)).all()):
            raise ValueError('Write positions outside ring')
        k=self.encoded_keys(symbols)
        v=self.value_encoder(values*2-1)
        mask=F.one_hot(positions,n).to(memory.dtype).unsqueeze(-1)
        if c.variant in ('fast','frozen_keys'):
            # Reuse the prior lab's checked B,H,W,V,K normalized delta primitive.
            A=memory.reshape(b,1,n,c.value_dim,c.key_dim)
            keys=k[:,None,None,:].expand(b,1,n,c.key_dim)
            vals=v[:,None,None,:].expand(b,1,n,c.value_dim)
            new=local_delta_write(A,keys,vals,write_mask=mask[:,None],rate=1.)
            return new.reshape_as(memory)
        inp=torch.cat([k,v],-1)[:,None,:].expand(b,n,-1)
        proposed=self.writer(inp.reshape(b*n,-1),memory.reshape(b*n,d)).reshape_as(memory)
        return memory+mask*(proposed-memory)

    def local_read(self, memory: Tensor, packet_key: Tensor) -> Tensor:
        b,n,d=memory.shape; c=self.config
        if packet_key.shape!=(b,n,c.key_dim): raise ValueError('Wrong key packet shape')
        if c.variant in ('fast','frozen_keys'):
            return torch.einsum('bnvk,bnk->bnv',memory.reshape(b,n,c.value_dim,c.key_dim),packet_key)
        # Exact-zero at an empty cell avoids a learned count-of-empty-cells bias.
        # This is supplied anchoring, documented as part of the control.
        zeros=torch.zeros_like(memory)
        return self.reader(torch.cat([memory,packet_key],-1))-self.reader(torch.cat([zeros,packet_key],-1))

    def query(self, memory: Tensor, symbols: Tensor, origins: Tensor,
              *, steps: int|None=None, communication: bool=True,
              return_trace: bool=False):
        """Local relay only; an output is issued at the origin on packet return.

        Before a complete lap the output at the origin contains no returning
        packet. Repeating laps would recount contributions and is rejected.
        """
        b,n,_=memory.shape; c=self.config
        if steps is None: steps=n
        if not 0<=steps<=n: raise ValueError('Query steps must be in [0, cells]')
        if origins.shape!=(b,) or symbols.shape!=(b,): raise ValueError('Wrong query shapes')
        if not bool(((origins>=0)&(origins<n)).all()): raise ValueError('Query origin outside ring')
        active=F.one_hot(origins,n).to(memory.dtype).unsqueeze(-1)
        key=self.encoded_keys(symbols)[:,None,:]*active
        acc=memory.new_zeros(b,n,c.value_dim)
        trace=[]
        for _ in range(steps):
            if communication:
                key=torch.roll(key,1,1);active=torch.roll(active,1,1);acc=torch.roll(acc,1,1)
            acc=acc+self.local_read(memory,key)*active
            if return_trace: trace.append({'valid':active.detach().clone(), 'acc':acc.detach().clone()})
        out=acc[torch.arange(b,device=memory.device),origins]
        logits=self.decoder(out)*self.log_scale.clamp(-2,5).exp()
        return (logits,trace) if return_trace else logits

    def parameter_counts(self):
        return {'total':sum(p.numel() for p in self.parameters()),
                'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'private_floats_per_cell':self.config.value_dim*self.config.key_dim,
                'packet_floats_per_cell':self.config.key_dim+self.config.value_dim+1}
    def config_dict(self): return asdict(self.config)


class ExactDictionary:
    """Non-neural positive control, same ring; exact one-hot addressing.

    With symbols=key_dim and bits=value_dim its numeric private payload has
    the same size as the fast model. Its keys are supplied orthogonal symbols.
    It is not a training result and has no learned weights.
    """
    def __init__(self,symbols=16,bits=8): self.symbols=symbols;self.bits=bits
    def initial_memory(self,batch,cells): return torch.zeros(batch,cells,self.symbols,self.bits)
    def write_event(self,memory,symbols,values,positions):
        out=memory.clone();out[torch.arange(len(symbols)),positions,symbols]=values*2-1
        return out
    def query(self,memory,symbols,origins):
        b,n,_,_=memory.shape
        key=F.one_hot(symbols,self.symbols).float()[:,None,:]
        active=F.one_hot(origins,n).float().unsqueeze(-1);key=key*active
        acc=torch.zeros(b,n,self.bits)
        for _ in range(n):
            active=torch.roll(active,1,1);key=torch.roll(key,1,1);acc=torch.roll(acc,1,1)
            acc+=torch.einsum('bnkv,bnk->bnv',memory,key)*active
        return acc[torch.arange(b),origins]
