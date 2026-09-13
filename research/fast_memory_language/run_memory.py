"""Train shared encoders through local fast writes; evaluate frozen parameters."""
from __future__ import annotations
import argparse,hashlib,json,time,sys,platform
from pathlib import Path
from dataclasses import asdict
import torch
import torch.nn.functional as F
from cellular_memory import CellularMemory,MemoryConfig,ExactDictionary
from tasks import sample,ingest

def fingerprint(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()): h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def score(logits,target):
    match=(logits>0)==(target>.5)
    return {'bit_accuracy':match.float().mean().item(), 'exact_accuracy':match.all(-1).float().mean().item()}

@torch.no_grad()
def evaluate(model,seed=91000,batch=512):
    before=fingerprint(model);model.eval();metrics={};examples=[]
    for name,cells,records,overwrite,concentrate in [
        ('in_distribution',8,8,False,False),('overwrite',8,8,True,False),
        ('more_records',8,16,False,False),('concentrated',8,16,False,True),
        ('larger_ring',16,8,False,False),('larger_ring_overwrite',16,16,True,False)]:
        ep=sample(batch,cells,records,torch.Generator().manual_seed(seed+cells*100+records*7+int(overwrite)),overwrite=overwrite,concentrate=concentrate)
        A=ingest(model,ep);ans=model.query(A,ep.query_symbol,ep.query_origin)
        result=score(ans,ep.target)
        result['state_rms']=A.double().square().mean().sqrt().item()
        metrics[name]=result
        if name=='in_distribution':
            metrics['reset_fast_memory']=score(model.query(torch.zeros_like(A),ep.query_symbol,ep.query_origin),ep.target)
            disabled=ingest(model,ep,enabled=False)
            metrics['writes_disabled']=score(model.query(disabled,ep.query_symbol,ep.query_origin),ep.target)
            metrics['communication_disabled']=score(model.query(A,ep.query_symbol,ep.query_origin,communication=False),ep.target)
            # Same addresses and keys, independent donor values. Full complement
            # makes original and donor targets maximally distinguishable.
            donor=sample(batch,cells,records,torch.Generator().manual_seed(1))
            donor.symbols=ep.symbols;donor.positions=ep.positions;donor.values=1-ep.values
            donorA=ingest(model,donor)
            donated=model.query(donorA,ep.query_symbol,ep.query_origin)
            metrics['transplant_vs_donor']=score(donated,1-ep.target)
            metrics['transplant_vs_original']=score(donated,ep.target)
            # Delete all memory at the query's source, not every cell.
            erased=A.clone();erased[torch.arange(batch),ep.source_position]=0
            metrics['erase_source_cell']=score(model.query(erased,ep.query_symbol,ep.query_origin),ep.target)
            metrics['insufficient_rounds']=score(model.query(A,ep.query_symbol,ep.query_origin,steps=cells-1),ep.target)
            for i in range(4):
                def bits(x):return ''.join(str(int(a)) for a in x)
                examples.append({'id':i,'symbol':ep.query_symbol[i].item(),'writer':ep.source_position[i].item(),
                    'reader':ep.query_origin[i].item(),'written':bits(ep.target[i]),'answer':bits(ans[i]>0),
                    'donor':bits(1-ep.target[i]),'transplanted_answer':bits(donated[i]>0)})
    E=model.encoded_keys(torch.arange(model.config.symbols));gram=E@E.T
    off=gram-torch.diag_embed(gram.diag())
    metrics['key_geometry']={'mean_abs_offdiag':off.abs().sum().item()/(len(E)*(len(E)-1)),
                            'max_abs_offdiag':off.abs().max().item()}
    assert fingerprint(model)==before,'Evaluation mutated shared parameters'
    return {'metrics':metrics,'examples':examples,'parameter_hash':before,'parameters_unchanged':True}


def run(variant,seed,steps,out,batch=64,log_every=100,key_dim=16):
    suffix=f'_k{key_dim}' if key_dim!=16 else ''
    path=out/f'{variant}{suffix}_seed{seed}';path.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);model=CellularMemory(MemoryConfig(variant=variant,key_dim=key_dim))
    init=fingerprint(model);optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.003,weight_decay=0.)
    g=torch.Generator().manual_seed(10000+seed);start=time.perf_counter()
    with (path/'training.jsonl').open('w') as f:
        for step in range(1,steps+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            ep=sample(batch,8,8,g,overwrite=(step%3==0))
            A=ingest(model,ep);logits=model.query(A,ep.query_symbol,ep.query_origin)
            loss=F.binary_cross_entropy_with_logits(logits,ep.target)
            if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at {step}')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            row={'step':step,'loss':loss.item(),**score(logits.detach(),ep.target),'gradient_norm':float(grad)}
            f.write(json.dumps(row)+'\n')
            if step%log_every==0 or step==1:print(variant,seed,row,flush=True)
    elapsed=time.perf_counter()-start
    result=evaluate(model);result.update({'seed':seed,'variant':variant,'training_steps':steps,'batch':batch,'seconds':elapsed,
             'counts':model.parameter_counts(),'initial_hash':init,'config':asdict(model.config),
             'status':'Exploratory CPU experiment; fixed ring transport, learned encodings'})
    torch.save({'model':model.state_dict(),'config':asdict(model.config)},path/'model.pt')
    loaded=torch.load(path/'model.pt',weights_only=True);clone=CellularMemory(MemoryConfig(**loaded['config']));clone.load_state_dict(loaded['model'])
    recheck=evaluate(clone)
    result['reload_matches']=recheck=={k:result[k] for k in ('metrics','examples','parameter_hash','parameters_unchanged')}
    assert result['reload_matches']
    (path/'metrics.json').write_text(json.dumps(result,indent=2))
    return result

@torch.no_grad()
def dictionary_results():
    model=ExactDictionary();result={}
    for name,n,r,ov,con in [('in_distribution',8,8,False,False),('overwrite',8,8,True,False),('concentrated',8,16,False,True),('larger_ring',16,8,False,False)]:
        ep=sample(512,n,r,torch.Generator().manual_seed(91000+n*100+r*7+int(ov)),overwrite=ov,concentrate=con)
        result[name]=score(model.query(ingest(model,ep),ep.query_symbol,ep.query_origin),ep.target)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--variants',nargs='+',default=['fast','frozen_keys','gru']);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--steps',type=int,default=600);p.add_argument('--batch',type=int,default=64);p.add_argument('--key-dim',type=int,default=16);p.add_argument('--out',type=Path,default=Path('results'));args=p.parse_args()
    if args.steps<1 or args.batch<1:p.error('steps and batch must be positive')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);args.out.mkdir(parents=True,exist_ok=True)
    environment={'torch':str(torch.__version__),'python':sys.version,'platform':platform.platform(),'device':'cpu','threads':torch.get_num_threads()}
    (args.out/'environment.json').write_text(json.dumps(environment,indent=2))
    for variant in args.variants:
        for seed in args.seeds:run(variant,seed,args.steps,args.out,args.batch,key_dim=args.key_dim)
    (args.out/'dictionary.json').write_text(json.dumps(dictionary_results(),indent=2))
if __name__=='__main__':main()
