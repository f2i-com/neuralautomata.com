"""Random association episodes. No query targets enter model methods."""
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class Episode:
    symbols: Tensor       # B,R observed symbol IDs
    values: Tensor        # B,R,bits legitimately observed values
    positions: Tensor     # B,R write sites; caller routes local observations
    query_symbol: Tensor  # B, no answer value
    query_origin: Tensor  # B, different from original write site
    target: Tensor        # B,bits, used by evaluator/loss only
    original_target: Tensor
    query_record: Tensor
    source_position: Tensor
    cells: int


def sample(batch:int,cells:int,records:int,g:torch.Generator,*,symbols=16,bits=8,
           overwrite:bool=False,concentrate:bool=False) -> Episode:
    if not 1<=records<=symbols or cells<2: raise ValueError('Invalid episode size')
    ids=torch.rand(batch,symbols,generator=g).argsort(-1)[:,:records]
    vals=torch.randint(0,2,(batch,records,bits),generator=g).float()
    pos=torch.randint(cells,(batch,records),generator=g)
    if concentrate: pos=pos[:,:1].expand_as(pos).clone()
    qi=torch.randint(records,(batch,),generator=g)
    bi=torch.arange(batch);q=ids[bi,qi];src=pos[bi,qi]
    origin=(src+torch.randint(1,cells,(batch,),generator=g))%cells
    target=vals[bi,qi].clone();old=target.clone()
    if overwrite:
        # Re-write the queried key at its original cell. This is NOT a global
        # version-resolution experiment involving conflicting writes elsewhere.
        new=1-target
        ids=torch.cat([ids,q[:,None]],1); vals=torch.cat([vals,new[:,None,:]],1)
        pos=torch.cat([pos,src[:,None]],1); target=new
    return Episode(ids,vals,pos,q,origin,target,old,qi,src,cells)


def ingest(model,ep:Episode,*,enabled=True):
    A=model.initial_memory(len(ep.symbols),ep.cells)
    for j in range(ep.symbols.shape[1]):
        if hasattr(model,'config'):
            A=model.write_event(A,ep.symbols[:,j],ep.values[:,j],ep.positions[:,j],enabled=enabled)
        else:
            A=model.write_event(A,ep.symbols[:,j],ep.values[:,j],ep.positions[:,j])
    return A
