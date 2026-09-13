import inspect
import torch
import torch.nn.functional as F
import pytest
from cellular_memory import CellularMemory,MemoryConfig,ExactDictionary
from tasks import sample,ingest
from causal_lm import CausalNCALM,LMConfig
from run_memory import fingerprint,score
from legacy.fast_memory import read,write

torch.set_num_threads(2)

def ep():return sample(4,8,8,torch.Generator().manual_seed(92))

def test_model_counts_and_capacity():
    m=CellularMemory(MemoryConfig());g=CellularMemory(MemoryConfig(variant='gru'))
    assert m.parameter_counts()['total']==385
    assert m.parameter_counts()['private_floats_per_cell']==g.parameter_counts()['private_floats_per_cell']==128

def test_single_write_is_local_and_nonmutating():
    m=CellularMemory(MemoryConfig());A=m.initial_memory(2,8);before=A.clone()
    new=m.write_event(A,torch.tensor([1,2]),torch.ones(2,8),torch.tensor([3,4]))
    assert torch.equal(A,before)
    mask=torch.ones(2,8,dtype=torch.bool);mask[0,3]=False;mask[1,4]=False
    assert torch.equal(new[mask],before[mask]);assert new.abs().sum()>0

def test_no_write_means_exact_identity():
    m=CellularMemory(MemoryConfig());e=ep();A=m.initial_memory(4,8)
    assert torch.equal(A,m.write_event(A,e.symbols[:,0],e.values[:,0],e.positions[:,0],enabled=False))

def test_query_does_not_change_private_memory_or_parameters():
    m=CellularMemory(MemoryConfig());e=ep();A=ingest(m,e);before=A.clone();weights=fingerprint(m)
    m.query(A,e.query_symbol,e.query_origin)
    assert torch.equal(A,before);assert weights==fingerprint(m)

def test_outer_gradient_reaches_keys_through_write():
    m=CellularMemory(MemoryConfig());e=ep();A=ingest(m,e)
    F.binary_cross_entropy_with_logits(m.query(A,e.query_symbol,e.query_origin),e.target).backward()
    assert m.keys.weight.grad is not None and torch.isfinite(m.keys.weight.grad).all()
    assert m.keys.weight.grad.abs().sum()>0

def test_frozen_keys_not_trainable():
    m=CellularMemory(MemoryConfig(variant='frozen_keys'));assert not m.keys.weight.requires_grad

def test_reset_memory_eliminates_all_episode_information():
    m=CellularMemory(MemoryConfig());e=ep();A=m.initial_memory(4,8)
    assert torch.equal(m.query(A,e.query_symbol,e.query_origin),torch.zeros(4,8))

def test_packet_moves_one_neighbor_per_step():
    m=CellularMemory(MemoryConfig());A=m.initial_memory(1,8)
    _,trace=m.query(A,torch.tensor([3]),torch.tensor([2]),return_trace=True)
    for t,record in enumerate(trace,1):
        expected=torch.zeros(1,8,1);expected[0,(2+t)%8,0]=1
        assert torch.equal(record['valid'],expected)

def test_no_return_before_complete_lap():
    m=CellularMemory(MemoryConfig());e=ep();A=ingest(m,e)
    assert torch.equal(m.query(A,e.query_symbol,e.query_origin,steps=7),torch.zeros(4,8))

def test_relay_agrees_with_sum_reference_after_full_lap():
    # Sum is an oracle for this fixed relay, NOT the implementation path.
    m=CellularMemory(MemoryConfig());e=ep();A=ingest(m,e)
    key=m.encoded_keys(e.query_symbol)[:,None,:].expand(4,8,-1)
    ref=m.decoder(m.local_read(A,key).sum(1))*m.log_scale.exp()
    assert torch.allclose(m.query(A,e.query_symbol,e.query_origin),ref,atol=1e-6)

def test_dictionary_stores_and_overwrites():
    e=sample(8,8,16,torch.Generator().manual_seed(6),overwrite=True,concentrate=True);m=ExactDictionary()
    assert score(m.query(ingest(m,e),e.query_symbol,e.query_origin),e.target)['exact_accuracy']==1.

def test_query_signature_does_not_accept_values_or_targets():
    names=inspect.signature(CellularMemory.query).parameters
    assert 'target' not in names and 'values' not in names

def test_batch_independence():
    m=CellularMemory(MemoryConfig());e=ep();A=ingest(m,e)
    before=m.query(A,e.query_symbol,e.query_origin);new=A.clone();new[0]=100
    after=m.query(new,e.query_symbol,e.query_origin)
    assert torch.equal(before[1:],after[1:])

def test_fast_memory_checkpoint_preserves_acquired_values(tmp_path):
    m=CellularMemory(MemoryConfig());e=ep();A=ingest(m,e).detach()
    path=tmp_path/'snapshot.pt';torch.save({'weights':m.state_dict(),'fast_memory':A},path)
    load=torch.load(path,weights_only=True);n=CellularMemory(MemoryConfig());n.load_state_dict(load['weights'])
    assert torch.equal(m.query(A,e.query_symbol,e.query_origin),n.query(load['fast_memory'],e.query_symbol,e.query_origin))

def test_delta_is_normalized_gradient_step():
    A=torch.randn(1,1,2,3,4,dtype=torch.float64,requires_grad=True)
    k=torch.randn(1,1,2,4,dtype=torch.float64);v=torch.randn(1,1,2,3,dtype=torch.float64)
    grad,=torch.autograd.grad(.5*(read(A,k)-v).square().sum(),A)
    ref=A-grad/(1e-8+k.square().sum(-1,keepdim=True)).unsqueeze(-1)
    assert torch.allclose(write(A,k,v),ref,atol=1e-12)

def test_language_prefix_invariance():
    torch.manual_seed(0);m=CausalNCALM(LMConfig(iterations=3))
    a=torch.randint(256,(2,20));b=a.clone();b[:,10:]=torch.randint(256,(2,10))
    assert torch.equal(m(a)[:,:10],m(b)[:,:10])

def test_language_streaming_equals_full_forward():
    m=CausalNCALM(LMConfig(iterations=4,radius=2));tokens=torch.randint(256,(2,21));cache=None;outs=[]
    for t in range(tokens.shape[1]):
        z,cache=m.stream_step(tokens[:,t],cache);outs.append(z)
    assert torch.allclose(torch.stack(outs,1),m(tokens),atol=2e-6,rtol=2e-6)

def test_language_remote_prefix_outside_lightcone_no_effect():
    m=CausalNCALM(LMConfig(iterations=2,radius=2));a=torch.randint(256,(1,20));b=a.clone();b[:,:15]=(b[:,:15]+1)%256
    assert m.max_context_bytes()==5
    assert torch.equal(m(a)[:,-1],m(b)[:,-1])

def test_language_training_gradients():
    m=CausalNCALM(LMConfig(iterations=2));x=torch.randint(256,(2,17))
    loss=F.cross_entropy(m(x[:,:-1]).reshape(-1,256),x[:,1:].reshape(-1));loss.backward()
    assert torch.isfinite(loss)
    assert m.perception.weight.grad is not None and m.perception.weight.grad.abs().sum()>0

def test_language_state_cache_bound():
    m=CausalNCALM(LMConfig(iterations=3,radius=2));cache=None
    for _ in range(10):_,cache=m.stream_step(torch.tensor([7]),cache)
    assert sum(x.numel() for x in cache)==3*2*32

@pytest.mark.parametrize('steps',[-1,9])
def test_invalid_extra_laps_rejected(steps):
    m=CellularMemory(MemoryConfig())
    with pytest.raises(ValueError):m.query(m.initial_memory(1,8),torch.tensor([1]),torch.tensor([2]),steps=steps)
