"""Reproduce acquired-memory replay and language sampling/depth checks."""
from pathlib import Path
from dataclasses import replace
import json
import torch
from cellular_memory import CellularMemory,MemoryConfig,ExactDictionary
from causal_lm import CausalNCALM,LMConfig,generate
from tasks import sample,ingest
from train_language import evaluate
from run_memory import score

@torch.no_grad()
def main():
    torch.set_num_threads(2)
    s=torch.load('results/fast_seed0/model.pt',weights_only=True);m=CellularMemory(MemoryConfig(**s['config']));m.load_state_dict(s['model']);m.eval()
    e=sample(4,8,8,torch.Generator().manual_seed(180071));A=ingest(m,e);before=m.query(A,e.query_symbol,e.query_origin)
    snap={'config':m.config_dict(),'weights':m.state_dict(),'private_memory':A,'query_symbol':e.query_symbol,'query_origin':e.query_origin}
    torch.save(snap,'results/acquired_memory_snapshot.pt')
    load=torch.load('results/acquired_memory_snapshot.pt',weights_only=True);n=CellularMemory(MemoryConfig(**load['config']));n.load_state_dict(load['weights'])
    after=n.query(load['private_memory'],load['query_symbol'],load['query_origin'])
    check={'identical_logits_after_reload':torch.equal(before,after),'targets':e.target.tolist(),'predictions':(after>0).int().tolist(),
           'note':'Evaluator targets are not part of the runtime snapshot.'}
    assert check['identical_logits_after_reload']
    Path('results/acquired_memory_snapshot_check.json').write_text(json.dumps(check,indent=2))
    d=ExactDictionary();ep=sample(512,8,16,torch.Generator().manual_seed(91912))
    dictionary=json.loads(Path('results/dictionary.json').read_text())
    dictionary['more_records']=score(d.query(ingest(d,ep),ep.query_symbol,ep.query_origin),ep.target)
    Path('results/dictionary.json').write_text(json.dumps(dictionary,indent=2))
    saved=torch.load('language_toy_longer/model.pt',weights_only=True);cfg=LMConfig(**saved['config']);lm=CausalNCALM(cfg);lm.load_state_dict(saved['model']);lm.eval()
    out={str(t):generate(lm,b'The ',160,temperature=t,seed=0).decode('utf-8',errors='replace') for t in [.3,.7,1.]}
    Path('language_toy_longer/temperature_samples.json').write_text(json.dumps(out,indent=2))
    data=torch.tensor(list(Path('toy_text/validation.txt').read_bytes()),dtype=torch.long)
    x=data[:128][None,:];cache=None;outs=[]
    for j in range(x.shape[1]):z,cache=lm.stream_step(x[:,j],cache);outs.append(z)
    full=lm(x);stream=torch.stack(outs,1)
    parity={'max_logit_abs_difference':(full-stream).abs().max().item(),'argmax_equal':bool(torch.equal(full.argmax(-1),stream.argmax(-1)))}
    Path('language_toy_longer/cache_parity.json').write_text(json.dumps(parity,indent=2))
    depths=[]
    for steps in [1,2,4,8,16,32]:
        model=CausalNCALM(replace(cfg,iterations=steps));model.load_state_dict(saved['model']);model.eval()
        depths.append({'iterations':steps,'max_context_bytes':model.max_context_bytes(),**evaluate(model,data,96)})
    Path('language_toy_longer/iteration_sweep.json').write_text(json.dumps(depths,indent=2))
    print(json.dumps(depths,indent=2))
if __name__=='__main__':main()
